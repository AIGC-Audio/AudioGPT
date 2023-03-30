from tkinter.tix import X_REGION
from numpy import isin
import torch
import torch.nn as nn
from audio_to_face.modules.audio2motion.transformer_base import *

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm='ln'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size
            if kernel_size is not None else 9,
            padding='SAME',
            norm=norm, act='gelu'
            )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


######################
# fastspeech modules
######################
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class FFTBlocks(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=None,
                 num_heads=2, use_pos_embed=True, use_last_norm=True, norm='ln',
                 use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout if dropout is not None else 0.1
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads,
                                    norm=norm)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
            elif norm == 'gn':
                self.layer_norm = GroupNorm1DTBC(8, embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
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
        return x

class SequentialSA(nn.Module):
    def __init__(self,layers):
        super(SequentialSA,self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self,x,x_mask):
        """
        x: [batch, T, H]
        x_mask: [batch, T]
        """
        pad_mask = 1. - x_mask
        for layer in self.layers:
            if isinstance(layer, EncSALayer):
                x = x.permute(1,0,2)
                x = layer(x,pad_mask)
                x = x.permute(1,0,2)
            elif isinstance(layer, nn.Linear):
                x = layer(x) * x_mask.unsqueeze(2)
            elif isinstance(layer, nn.AvgPool1d):
                x = x.permute(0,2,1)
                x = layer(x)
                x = x.permute(0,2,1)
            elif isinstance(layer, nn.PReLU):
                bs, t, hid = x.shape
                x = x.reshape([bs*t,hid])
                x = layer(x)
                x = x.reshape([bs, t, hid])
            else: # Relu
                x = layer(x) 
            
        return x

class TransformerStyleFusionModel(nn.Module):
    def __init__(self, num_heads=4, dropout = 0.1, out_dim = 64):
        super(TransformerStyleFusionModel, self).__init__()
        self.audio_layer = SequentialSA([
            nn.Linear(29, 48),
            nn.ReLU(48),
            nn.Linear(48, 128),
        ])

        self.energy_layer = SequentialSA([
            nn.Linear(1, 16),
            nn.ReLU(16),
            nn.Linear(16, 64),
        ])

        self.backbone1 = FFTBlocks(hidden_size=192,num_layers=3)

        self.sty_encoder = nn.Sequential(*[
            nn.Linear(135, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ])

        self.backbone2 = FFTBlocks(hidden_size=320,num_layers=3)

        self.out_layer = SequentialSA([
            nn.AvgPool1d(kernel_size=2,stride=2,padding=0), #[b,hid,t_audio]=>[b,hid,t_audio//2]
            nn.Linear(320,out_dim),
            nn.PReLU(out_dim),
            nn.Linear(out_dim,out_dim),
        ])

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, audio, energy, style, x_mask, y_mask):
        pad_mask = 1. - x_mask
        audio_feat = self.audio_layer(audio, x_mask)
        energy_feat = self.energy_layer(energy, x_mask)
        feat = torch.cat((audio_feat, energy_feat), dim=-1) # [batch, T, H=48+16]
        feat = self.backbone1(feat, pad_mask)
        feat = self.dropout(feat)

        sty_feat = self.sty_encoder(style) # [batch,135]=>[batch, H=64]
        sty_feat = sty_feat.unsqueeze(1).repeat(1, feat.shape[1], 1) # [batch, T, H=64]

        feat = torch.cat([feat, sty_feat], dim=-1) # [batch, T, H=64+64]
        feat = self.backbone2(feat, pad_mask) # [batch, T, H=128]
        out = self.out_layer(feat, y_mask) # [batch, T//2, H=out_dim]

        return out


if __name__ == '__main__':
    model = TransformerStyleFusionModel()
    audio = torch.rand(4,200,29) # [B,T,H]
    energy = torch.rand(4,200,1) # [B,T,H]
    style = torch.ones(4,135) # [B,T]
    x_mask = torch.ones(4,200) # [B,T]
    x_mask[3,10:] = 0
    ret = model(audio,energy,style, x_mask)
    print(" ")