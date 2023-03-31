from modules.fastspeech.tts_modules import FastspeechDecoder
# from modules.fastspeech.fast_tacotron import DecoderRNN
# from modules.fastspeech.speedy_speech.speedy_speech import ConvBlocks
# from modules.fastspeech.conformer.conformer import ConformerDecoder
import torch
from torch.nn import functional as F
import torch.nn as nn
import math
from utils.hparams import hparams
from .diffusion import Mish
Linear = nn.Linear


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class FFT(FastspeechDecoder):
    def __init__(self, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None):
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)
        dim = hparams['residual_channels']
        self.input_projection = Conv1d(hparams['audio_num_mel_bins'], dim, 1)
        self.diffusion_embedding = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.get_mel_out = Linear(hparams['hidden_size'], 80, bias=True)
        self.get_decode_inp = Linear(hparams['hidden_size'] + dim + dim,
                                     hparams['hidden_size'])  # hs + dim + 80 -> hs

    def forward(self, spec, diffusion_step, cond, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param spec: [B, 1, 80, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x).permute([0, 2, 1])  #  [B, T, residual_channel]
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)  # [B, dim]
        cond = cond.permute([0, 2, 1])  # [B, T, M]

        seq_len = cond.shape[1]  # [T_mel]
        time_embed = diffusion_step[:, None, :]  # [B, 1, dim]
        time_embed = time_embed.repeat([1, seq_len, 1])  # # [B, T, dim]

        decoder_inp = torch.cat([x, cond, time_embed], dim=-1)  # [B, T, dim + H + dim]
        decoder_inp = self.get_decode_inp(decoder_inp)  # [B, T, H]
        x = decoder_inp

        '''
        Required x: [B, T, C]
        :return: [B, T, C] or [L, B, T, C]
        '''
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]

        x = self.get_mel_out(x).permute([0, 2, 1])  # [B, 80, T]
        return x[:, None, :, :]