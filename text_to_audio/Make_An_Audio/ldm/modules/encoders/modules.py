import torch
import torch.nn as nn
from functools import partial

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, AutoTokenizer
from importlib_resources import files
from ldm.modules.encoders.CLAP.utils import read_config_as_args
from ldm.modules.encoders.CLAP.clap import TextEncoder
from ldm.util import default, count_params
import open_clip

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]# (bsz,1)
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):# 这里不是用的pretrained bert,是用的transformers的BertTokenizer加自定义的TransformerWrapper
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLAPEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text (from huggingface)"""
    def __init__(self, weights_path, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        # To device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )

        self.max_length = max_length
        self.device = device
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False


    def encode(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.caption_encoder.base(input_ids=tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        return z

class FrozenCLAPEmbedderNoLoad(AbstractEncoder):
    def __init__(self, config, freeze=True, device="cpu", max_length=77):
        super().__init__()
        args = config

        # To device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )

        self.max_length = max_length
        self.device = device
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")        

    def freeze(self):
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False


    def encode(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.caption_encoder.base(input_ids=tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        return z


class NewFrozenCLAPEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text (from huggingface)"""
    def __init__(self, weights_path, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        # To device
        from transformers import RobertaTokenizer
        from ldm.modules.encoders.open_clap import create_model


        model, model_cfg = create_model(
            'HTSAT-tiny',
            'roberta',
            weights_path,
            enable_fusion=True,
            fusion_type='aff_2d'
        )

        del model.audio_branch, model.audio_transform, model.audio_projection
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = model

        self.max_length = max_length
        self.device = device
        if freeze: self.freeze()

        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{self.model.__class__.__name__} comes with: {param_num / 1e+6:.3f} M params.')

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        outputs = self.model.text_branch(input_ids=batch_encoding["input_ids"].to(self.device), attention_mask=batch_encoding["attention_mask"].to(self.device))
        z = self.model.text_projection(outputs.last_hidden_state)
        return z

class FrozenFLANEmbedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/flan-t5-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
class FrozenGlobalNormOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", freeze=True, delvisual=True):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        if delvisual:
            del model.visual
            del preprocess
        else:
            self.preprocess = preprocess
        self.model = model

        self.device = device
        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.model.encode_text(tokens.to(self.device))
        z /= z.norm(dim=-1, keepdim=True)
        return z.unsqueeze(1)

    def forward_img(self, image):
        z = self.model.encode_image(image.to(self.device))
        z /= z.norm(dim=-1, keepdim=True)
        return z.unsqueeze(1)

    def encode(self, text):
        return self(text)