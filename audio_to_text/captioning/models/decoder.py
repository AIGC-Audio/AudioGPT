# -*- coding: utf-8 -*-

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .utils import generate_length_mask, init, PositionalEncoding


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    Base decoder, cannot be called directly
    All decoders should inherit from this class
    """

    def __init__(self, emb_dim, vocab_size, fc_emb_dim,
                 attn_emb_dim, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.fc_emb_dim = fc_emb_dim
        self.attn_emb_dim = attn_emb_dim
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.in_dropout = nn.Dropout(dropout)

    def forward(self, x):
        raise NotImplementedError

    def load_word_embedding(self, weight, freeze=True):
        embedding = np.load(weight)
        assert embedding.shape[0] == self.vocab_size, "vocabulary size mismatch"
        assert embedding.shape[1] == self.emb_dim, "embed size mismatch"
        
        # embeddings = torch.as_tensor(embeddings).float()
        # self.word_embeddings.weight = nn.Parameter(embeddings)
        # for para in self.word_embeddings.parameters():
            # para.requires_grad = tune
        self.word_embedding = nn.Embedding.from_pretrained(embedding,
            freeze=freeze)


class RnnDecoder(BaseDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout,)
        self.d_model = d_model
        self.num_layers = kwargs.get('num_layers', 1)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.rnn_type = kwargs.get('rnn_type', "GRU")
        self.classifier = nn.Linear(
            self.d_model * (self.bidirectional + 1), vocab_size)

    def forward(self, x):
        raise NotImplementedError

    def init_hidden(self, bs, device):
        num_dire = self.bidirectional + 1
        n_layer = self.num_layers
        hid_dim = self.d_model
        if self.rnn_type == "LSTM":
            return (torch.zeros(num_dire * n_layer, bs, hid_dim).to(device),
                    torch.zeros(num_dire * n_layer, bs, hid_dim).to(device))
        else:
            return torch.zeros(num_dire * n_layer, bs, hid_dim).to(device)


class RnnFcDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 2,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.apply(init)
    
    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None)
        fc_emb = input_dict["fc_emb"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))
        
        p_fc_emb = self.fc_proj(fc_emb)
        # embed: [N, T, embed_size]
        embed = torch.cat((embed, p_fc_emb), dim=-1)

        out, state = self.model(embed, state)
        # out: [N, T, hs], states: [num_layers * num_dire, N, hs]
        logits = self.classifier(out)
        output = {
            "state": state,
            "embeds": out,
            "logits": logits
        }

        return output


class Seq2SeqAttention(nn.Module):

    def __init__(self, hs_enc, hs_dec, attn_size):
        """
        Args:
            hs_enc: encoder hidden size
            hs_dec: decoder hidden size
            attn_size: attention vector size
        """
        super(Seq2SeqAttention, self).__init__()
        self.h2attn = nn.Linear(hs_enc + hs_dec, attn_size)
        self.v = nn.Parameter(torch.randn(attn_size))
        self.apply(init)

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden (query), [N, hs_dec]
            h_enc: encoder memory (key/value), [N, src_max_len, hs_enc]
            src_lens: source (encoder memory) lengths, [N, ]
        """
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        score = torch.bmm(v, attn_out.transpose(1, 2)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]

        return ctx, weights


class AttentionProj(nn.Module):

    def __init__(self, hs_enc, hs_dec, embed_dim, attn_size):
        self.q_proj = nn.Linear(hs_dec, embed_dim)
        self.kv_proj = nn.Linear(hs_enc, embed_dim)
        self.h2attn = nn.Linear(embed_dim * 2, attn_size)
        self.v = nn.Parameter(torch.randn(attn_size))
        self.apply(init)

    def init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden (query), [N, hs_dec]
            h_enc: encoder memory (key/value), [N, src_max_len, hs_enc]
            src_lens: source (encoder memory) lengths, [N, ]
        """
        h_enc = self.kv_proj(h_enc) # [N, src_max_len, embed_dim]
        h_dec = self.q_proj(h_dec) # [N, embed_dim]
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        score = torch.bmm(v, attn_out.transpose(1, 2)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]

        return ctx, weights


class BahAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                             self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_fc_emb = self.fc_proj(fc_emb)
        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_emb.unsqueeze(1)),
                              dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class BahAttnDecoder2(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        add fc, attn, word together to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                             self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.attn_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.apply(partial(init, method="xavier"))

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))
        p_attn_emb = self.attn_proj(attn_emb)

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, p_attn_emb, attn_emb_len)

        p_fc_emb = self.fc_proj(fc_emb)
        rnn_input = embed + c.unsqueeze(1) + p_fc_emb.unsqueeze(1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class ConditionalBahAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                             self.num_layers,
                                     attn_size)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.condition_embedding = nn.Embedding(2, emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        condition = input_dict["condition"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))

        condition = torch.as_tensor([[1 - c, c] for c in condition]).to(fc_emb.device)
        condition_emb = torch.matmul(condition, self.condition_embedding.weight)
        # condition_embs: [N, emb_dim]

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), condition_emb.unsqueeze(1)),
                              dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class StructBahAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, struct_vocab_size,
                 attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                         self.num_layers,
                                     attn_size)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.struct_embedding = nn.Embedding(struct_vocab_size, emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        structure = input_dict["structure"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))

        struct_emb = self.struct_embedding(structure)
        # struct_embs: [N, emb_dim]

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), struct_emb.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class StyleBahAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                        self.num_layers,
                                     attn_size)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        style = input_dict["style"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), style.unsqueeze(1)),
                              dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class BahAttnDecoder3(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                         self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]

        if word.size(-1) == self.fc_emb_dim: # fc_emb
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_emb.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class SpecificityBahAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim + 1,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                         self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        condition = input_dict["condition"] # [N,]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat(
            (embed, p_ctx.unsqueeze(1), condition.reshape(-1, 1, 1)),
            dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class TransformerDecoder(BaseDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout=dropout,)
        self.d_model = emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerDecoder(layer, self.nlayers)
        self.classifier = nn.Linear(self.d_model, vocab_size)
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_emb_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model)
        )
        # self.attn_proj = lambda x: x
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output




class EventTransformerDecoder(TransformerDecoder):

    def forward(self, input_dict):
        word = input_dict["word"] # index of word embeddings
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]
        event_emb = input_dict["event"] # [N, emb_dim]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]

        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed += event_emb
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output


class KeywordProbTransformerDecoder(TransformerDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, keyword_classes_num, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, **kwargs)
        self.keyword_proj = nn.Linear(keyword_classes_num, self.d_model)
        self.word_keyword_norm = nn.LayerNorm(self.d_model)

    def forward(self, input_dict):
        word = input_dict["word"] # index of word embeddings
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]
        keyword = input_dict["keyword"] # [N, keyword_classes_num]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]

        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed += self.keyword_proj(keyword)
        embed = self.word_keyword_norm(embed)

        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output
