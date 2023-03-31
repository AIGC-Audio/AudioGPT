import math
import torch
from torch import nn
from torch.nn import functional as F
from utils.hparams import hparams
from modules.commons.common_layers import Embedding
from utils.tts_utils import group_hidden_by_segs, expand_word2ph

import transformers

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.,
                 window_size=None, block_length=None, pre_ln=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.pre_ln = pre_ln

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size,
                                   p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))
        if pre_ln:
            self.last_ln = LayerNorm(hidden_channels)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            x_ = x
            if self.pre_ln:
                x = self.norm_layers_1[i](x)
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = x_ + y
            if not self.pre_ln:
                x = self.norm_layers_1[i](x)

            x_ = x
            if self.pre_ln:
                x = self.norm_layers_2[i](x)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = x_ + y
            if not self.pre_ln:
                x = self.norm_layers_2[i](x)
        if self.pre_ln:
            x = self.last_ln(x)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0.,
                 block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class RelTransformerEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout=0.0,
                 window_size=4,
                 block_length=None,
                 prenet=True,
                 pre_ln=True,
                 ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        if n_vocab > 0:
            self.emb = Embedding(n_vocab, hidden_channels, padding_idx=0)

        if prenet:
            self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels,
                                    kernel_size=5, n_layers=3, p_dropout=0)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

    def forward(self, x, x_mask=None):
        if self.n_vocab > 0:
            x_lengths = (x > 0).long().sum(-1)
            x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        else:
            x_lengths = (x.abs().sum(-1) > 0).long().sum(-1)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)
        return x.transpose(1, 2)


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y):
        sim = self.cos(x, y)
        self.record = sim.detach() # [64,64]
        min_size = min(self.record.shape[0], self.record.shape[1]) # 64
        num_item = self.record.shape[0] * self.record.shape[1] # 4096
        self.pos_avg = self.record.diag().sum() / min_size
        if num_item - min_size == 0:
            self.neg_avg = (self.record.sum() - self.record.diag().sum()) / 1
            return sim / self.temp
        if torch.any(torch.isnan(self.record)).item() is True:
            print("we got self.record has nan when compute neg_avg")
        if torch.any(torch.isnan(self.record.diag())).item() is True:
            print("we got self.record.diag() has nan when compute neg_avg")
        self.neg_avg = (self.record.sum() - self.record.diag().sum()) / (num_item - min_size)
            
        return sim / self.temp


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hid_dim)
        self.decoder = nn.Linear(hid_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# V2_2
# change add to concat.
# now support finetune BERT
# grad_bert=0.1 & trainable_block_idx=0
class BERTRelTransformerEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout=0.0,
                 window_size=4,
                 block_length=None,
                 prenet=True,
                 pre_ln=True,
                 ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        if n_vocab > 0:
            self.emb = Embedding(n_vocab, hidden_channels, padding_idx=0)

        if prenet:
            self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels,
                                    kernel_size=5, n_layers=3, p_dropout=0)
        self.encoder1 = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers//2,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

        self.encoder2 = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers - n_layers//2,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

        if hparams['ds_name'] in ['ljspeech', 'libritts', 'librispeech']:
            model_name = 'bert-base-uncased'
        elif hparams['ds_name'] in ['biaobei', 'wenetspeech']:
            model_name = 'bert-base-chinese'
        else:
            raise NotImplementedError()
            
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        if hparams.get("load_bert_from_pretrained", True):
            print("Load BERT from pretrained model ...")
            self.bert =  transformers.AutoModel.from_pretrained(model_name,config=config)
            trainable_start_block = hparams.get("bert_trainable_start_block", 0)
        else:
            print("Initialize BERT from scratch!")
            self.bert =  transformers.BertModel(config=config)
            trainable_start_block = 0
        
        for k, v in self.bert.named_parameters():
            if 'embeddings' in k:
                v.requires_grad = False
            elif 'encoder.layer' in k:
                block_idx =  int(k.split(".")[2])
                if block_idx < trainable_start_block:
                    v.requires_grad = False
                else:
                    v.requires_grad = True
            elif 'cls' in k:
                v.requires_grad = True
            else:
                print("Unhandled key: {}, set to requires_grad...".format(k))
                v.requires_grad = True

        self.bert_combine = nn.Sequential(*[
            nn.Conv1d(768 + hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            ])
        self.pooler = Pooler("avg")
        self.sim = Similarity(temp=0.05)
        
    def forward(self, x, x_mask=None, bert_feats=None, ph2word=None, **kwargs):
        if self.n_vocab > 0:
            x_lengths = (x > 0).long().sum(-1)
            x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        else:
            x_lengths = (x.abs().sum(-1) > 0).long().sum(-1)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder1(x, x_mask)
        bert_outputs = self.bert(bert_feats['bert_input_ids'],
                                    attention_mask=bert_feats['bert_attention_mask'],
                                    token_type_ids=bert_feats['bert_token_type_ids'],
                                    output_hidden_states=True)
        bert_num_blocks = hparams.get("bert_num_blocks", 12) # total 1+12blocks in bert
        bert_embedding = bert_outputs['hidden_states'][bert_num_blocks]
        # bert_embedding = bert_outputs['last_hidden_state']
        grad_bert = hparams.get("grad_bert", 0.1)
        bert_embedding = bert_embedding.detach() * (1-grad_bert) + bert_embedding * grad_bert
        bert_word_embedding, _ = group_hidden_by_segs(bert_embedding, bert_feats['bert_token2word'], bert_feats['bert_token2word'].max().item())
        bert_ph_embedding = expand_word2ph(bert_word_embedding, ph2word)
        bert_ph_embedding = bert_ph_embedding.transpose(1,2)
        x = torch.cat([x, bert_ph_embedding], dim=1)
        x = self.bert_combine(x)
        x = self.encoder2(x, x_mask)
        return x.transpose(1, 2)


