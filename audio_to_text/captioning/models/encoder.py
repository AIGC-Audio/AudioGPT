# -*- coding: utf-8 -*-

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torchlibrosa.augmentation import SpecAugmentation

from .utils import mean_with_lens, max_with_lens, \
    init, pack_wrapper, generate_length_mask, PositionalEncoding


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class BaseEncoder(nn.Module):
    
    """
    Encode the given audio into embedding
    Base encoder class, cannot be called directly
    All encoders should inherit from this class
    """

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim):
        super(BaseEncoder, self).__init__()
        self.spec_dim = spec_dim
        self.fc_feat_dim = fc_feat_dim
        self.attn_feat_dim = attn_feat_dim


    def forward(self, x):
        #########################
        # an encoder first encodes audio feature into embedding, obtaining
        # `encoded`: {
        #     fc_embs: [N, fc_emb_dim],
        #     attn_embs: [N, attn_max_len, attn_emb_dim],
        #     attn_emb_lens: [N,]
        # }
        #########################
        raise NotImplementedError


class Block2D(nn.Module):

    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class AttentionPool(nn.Module):  
    """docstring for AttentionPool"""  
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T, D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


class MMPool(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.avgpool = nn.AvgPool2d(dims)
        self.maxpool = nn.MaxPool2d(dims)

    def forward(self, x):
        return self.avgpool(x) + self.maxpool(x)


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1
    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':  
        return AttentionPool(inputdim=kwargs['inputdim'],  
                             outputdim=kwargs['outputdim'])


def embedding_pooling(x, lens, pooling="mean"):
    if pooling == "max":
        fc_embs = max_with_lens(x, lens)
    elif pooling == "mean":
        fc_embs = mean_with_lens(x, lens)
    elif pooling == "mean+max":
        x_mean = mean_with_lens(x, lens)
        x_max = max_with_lens(x, lens)
        fc_embs = x_mean + x_max
    elif pooling == "last":
        indices = (lens - 1).reshape(-1, 1, 1).repeat(1, 1, x.size(-1))
        # indices: [N, 1, hidden]
        fc_embs = torch.gather(x, 1, indices).squeeze(1)
    else:
        raise Exception(f"pooling method {pooling} not support")
    return fc_embs


class Cdur5Encoder(BaseEncoder):

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim, pooling="mean"):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.pooling = pooling
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(
                torch.randn(1, 1, 500, spec_dim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.apply(init)

    def forward(self, input_dict):
        x = input_dict["spec"]
        lens = input_dict["spec_len"]
        if "upsample" not in input_dict:
            input_dict["upsample"] = False
        lens = torch.as_tensor(copy.deepcopy(lens))
        N, T, _ = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        if input_dict["upsample"]:
            x = nn.functional.interpolate(
                x.transpose(1, 2),
                T,
                mode='linear',
                align_corners=False).transpose(1, 2)
        else:
            lens //= 4
        attn_emb = x
        fc_emb = embedding_pooling(x, lens, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }


def conv_conv_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=3,
                  bias=False,
                  padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True),
        nn.Conv2d(out_channel,
                  out_channel,
                  kernel_size=3,
                  bias=False,
                  padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True)
    )


class Cdur8Encoder(BaseEncoder):
    
    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim, pooling="mean"):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.pooling = pooling
        self.features = nn.Sequential(
            conv_conv_block(1, 64),
            MMPool((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(64, 128),
            MMPool((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(128, 256),
            MMPool((1, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(256, 512),
            MMPool((1, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.init_bn = nn.BatchNorm2d(spec_dim)
        self.embedding = nn.Linear(512, 512)
        self.gru = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.apply(init)

    def forward(self, input_dict):
        x = input_dict["spec"]
        lens = input_dict["spec_len"]
        lens = torch.as_tensor(copy.deepcopy(lens))
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = x.transpose(1, 3)
        x = self.init_bn(x)
        x = x.transpose(1, 3)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.embedding(x))
        x, _ = self.gru(x)
        attn_emb = x
        lens //= 4
        fc_emb = embedding_pooling(x, lens, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }


class Cnn10Encoder(BaseEncoder):

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.features = nn.Sequential(
            conv_conv_block(1, 64),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(64, 128),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(128, 256),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(256, 512),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.init_bn = nn.BatchNorm2d(spec_dim)
        self.embedding = nn.Linear(512, 512)
        self.apply(init)

    def forward(self, input_dict):
        x = input_dict["spec"]
        lens = input_dict["spec_len"]
        lens = torch.as_tensor(copy.deepcopy(lens))
        x = x.unsqueeze(1)  # [N, 1, T, D]
        x = x.transpose(1, 3)
        x = self.init_bn(x)
        x = x.transpose(1, 3)
        x = self.features(x) # [N, 512, T/16, 1]
        x = x.transpose(1, 2).contiguous().flatten(-2) # [N, T/16, 512]
        attn_emb = x
        lens //= 16
        fc_emb = embedding_pooling(x, lens, "mean+max")
        fc_emb = F.dropout(fc_emb, p=0.5, training=self.training)
        fc_emb = self.embedding(fc_emb)
        fc_emb = F.relu_(fc_emb)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14Encoder(nn.Module):
    def __init__(self, sample_rate=32000):
        super().__init__()
        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def load_pretrained(self, pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "model" in checkpoint:
            state_keys = checkpoint["model"].keys()
            backbone = False
            for key in state_keys:
                if key.startswith("backbone."):
                    backbone = True
                    break

            if backbone: # COLA
                state_dict = {}
                for key, value in checkpoint["model"].items():
                    if key.startswith("backbone."):
                        model_key = key.replace("backbone.", "")
                        state_dict[model_key] = value
            else: # PANNs
                state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint: # CLAP
            state_dict = checkpoint["state_dict"]
            state_dict_keys = list(filter(
                lambda x: "audio_encoder" in x, state_dict.keys()))
            state_dict = {
                key.replace('audio_encoder.', ''): state_dict[key]
                    for key in state_dict_keys
            }
        else:
            raise Exception("Unkown checkpoint format")

        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() if (k in model_dict) and (
                model_dict[k].shape == v.shape)
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=True)
 
    def forward(self, input_dict):
        """
        Input: (batch_size, n_samples)"""
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(attn_emb, feat_length)
        x_mean = mean_with_lens(attn_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        fc_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }

        return output_dict


class RnnEncoder(BaseEncoder):

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim,
                 pooling="mean", **kwargs):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.pooling = pooling
        self.hidden_size = kwargs.get('hidden_size', 512)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.2)
        self.rnn_type = kwargs.get('rnn_type', "GRU")
        self.in_bn = kwargs.get('in_bn', False)
        self.embed_dim = self.hidden_size * (self.bidirectional + 1)
        self.network = getattr(nn, self.rnn_type)(
            attn_feat_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        if self.in_bn:
            self.bn = nn.BatchNorm1d(self.embed_dim)
        self.apply(init)

    def forward(self, input_dict):
        x = input_dict["attn"]
        lens = input_dict["attn_len"]
        lens = torch.as_tensor(lens)
        # x: [N, T, E]
        if self.in_bn:
            x = pack_wrapper(self.bn, x, lens)
        out = pack_wrapper(self.network, x, lens)
        # out: [N, T, hidden]
        attn_emb = out
        fc_emb = embedding_pooling(out, lens, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }


class Cnn14RnnEncoder(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None,
                 freeze_cnn=False, freeze_cnn_bn=False,
                 pooling="mean", **kwargs):
        super().__init__()
        self.cnn = Cnn14Encoder(sample_rate)
        self.rnn = RnnEncoder(64, 2048, 2048, pooling, **kwargs)
        if pretrained is not None:
            self.cnn.load_pretrained(pretrained)
        if freeze_cnn:
            assert pretrained is not None, "cnn is not pretrained but frozen"
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.freeze_cnn_bn = freeze_cnn_bn

    def train(self, mode):
        super().train(mode=mode)
        if self.freeze_cnn_bn:
            def bn_eval(module):
                class_name = module.__class__.__name__
                if class_name.find("BatchNorm") != -1:
                    module.eval()
            self.cnn.apply(bn_eval)
        return self

    def forward(self, input_dict):
        output_dict = self.cnn(input_dict)
        output_dict["attn"] = output_dict["attn_emb"]
        output_dict["attn_len"] = output_dict["attn_emb_len"]
        del output_dict["attn_emb"], output_dict["attn_emb_len"]
        output_dict = self.rnn(output_dict)
        return output_dict


class TransformerEncoder(BaseEncoder):

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim, d_model, **kwargs):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.d_model = d_model
        dropout = kwargs.get("dropout", 0.2)
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.attn_proj = nn.Sequential(
            nn.Linear(attn_feat_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model)
        )
        layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerEncoder(layer, self.nlayers)
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_dict):
        attn_feat = input_dict["attn"]
        attn_feat_len = input_dict["attn_len"]
        attn_feat_len = torch.as_tensor(attn_feat_len)

        attn_feat = self.attn_proj(attn_feat) # [bs, T, d_model]

        cls_emb = self.cls_token.reshape(1, 1, self.d_model).repeat(
            attn_feat.size(0), 1, 1)
        attn_feat = torch.cat((cls_emb, attn_feat), dim=1)
        attn_feat = attn_feat.transpose(0, 1)

        attn_feat_len += 1
        src_key_padding_mask = ~generate_length_mask(
            attn_feat_len, attn_feat.size(0)).to(attn_feat.device)
        output = self.model(attn_feat, src_key_padding_mask=src_key_padding_mask)

        attn_emb = output.transpose(0, 1)
        fc_emb = attn_emb[:, 0]
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": attn_feat_len
        }


class Cnn14TransformerEncoder(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None,
                 freeze_cnn=False, freeze_cnn_bn=False,
                 d_model="mean", **kwargs):
        super().__init__()
        self.cnn = Cnn14Encoder(sample_rate)
        self.trm = TransformerEncoder(64, 2048, 2048, d_model, **kwargs)
        if pretrained is not None:
            self.cnn.load_pretrained(pretrained)
        if freeze_cnn:
            assert pretrained is not None, "cnn is not pretrained but frozen"
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.freeze_cnn_bn = freeze_cnn_bn

    def train(self, mode):
        super().train(mode=mode)
        if self.freeze_cnn_bn:
            def bn_eval(module):
                class_name = module.__class__.__name__
                if class_name.find("BatchNorm") != -1:
                    module.eval()
            self.cnn.apply(bn_eval)
        return self

    def forward(self, input_dict):
        output_dict = self.cnn(input_dict)
        output_dict["attn"] = output_dict["attn_emb"]
        output_dict["attn_len"] = output_dict["attn_emb_len"]
        del output_dict["attn_emb"], output_dict["attn_emb_len"]
        output_dict = self.trm(output_dict)
        return output_dict





