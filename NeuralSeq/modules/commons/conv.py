import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import LayerNorm


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12):
        super(ResidualBlock, self).__init__()

        if norm_type == 'bn':
            norm_builder = lambda: nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm_builder = lambda: nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
        else:
            norm_builder = lambda: nn.Identity()

        self.blocks = [
            nn.Sequential(
                norm_builder(),
                nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
                          padding=(dilation * (kernel_size - 1)) // 2),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                nn.GELU(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
            )
            for i in range(n)
        ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return x


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, hidden_size, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5,
                 init_weights=True, is_BTC=True, num_layers=None, post_net_kernel=3):
        super(ConvBlocks, self).__init__()
        self.is_BTC = is_BTC
        if num_layers is not None:
            dilations = [1] * num_layers
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps)
              for d in dilations],
        )
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(hidden_size)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(hidden_size, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, hidden_size)
        elif norm_type == 'ln':
            norm = LayerNorm(hidden_size, dim=1, eps=ln_eps)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(hidden_size, out_dims, kernel_size=post_net_kernel,
                                   padding=post_net_kernel // 2)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x, nonpadding=None):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        if self.is_BTC:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class TextConvEncoder(ConvBlocks):
    def __init__(self, dict_size, hidden_size, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, num_layers=None, post_net_kernel=3):
        super().__init__(hidden_size, out_dims, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights, num_layers=num_layers,
                         post_net_kernel=post_net_kernel)
        self.embed_tokens = Embedding(dict_size, hidden_size, 0)
        self.embed_scale = math.sqrt(hidden_size)

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [B x T x C]
        }
        """
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        return super().forward(x)


class ConditionalConvBlocks(ConvBlocks):
    def __init__(self, hidden_size, c_cond, c_out, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, is_BTC=True, num_layers=None):
        super().__init__(hidden_size, c_out, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights, is_BTC=False, num_layers=num_layers)
        self.g_prenet = nn.Conv1d(c_cond, hidden_size, 3, padding=1)
        self.is_BTC_ = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, cond, nonpadding=None):
        if self.is_BTC_:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2)
            if nonpadding is not None:
                nonpadding = nonpadding.transpose(1, 2)
        if nonpadding is None:
            nonpadding = x.abs().sum(1)[:, None]
        x = x + self.g_prenet(cond)
        x = x * nonpadding
        x = super(ConditionalConvBlocks, self).forward(x)  # input needs to be BTC
        if self.is_BTC_:
            x = x.transpose(1, 2)
        return x
