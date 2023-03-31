import torch
from torch import nn
from modules.commons.conv import ConditionalConvBlocks
from modules.commons.wavenet import WN


class FlipLayer(nn.Module):
    def forward(self, x, nonpadding, cond=None, reverse=False):
        x = torch.flip(x, [1])
        return x


class CouplingLayer(nn.Module):
    def __init__(self, c_in, hidden_size, kernel_size, n_layers, p_dropout=0, c_in_g=0, nn_type='wn'):
        super().__init__()
        self.channels = c_in
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.c_half = c_in // 2

        self.pre = nn.Conv1d(self.c_half, hidden_size, 1)
        if nn_type == 'wn':
            self.enc = WN(hidden_size, kernel_size, 1, n_layers, p_dropout=p_dropout,
                          c_cond=c_in_g)
        elif nn_type == 'conv':
            self.enc = ConditionalConvBlocks(
                hidden_size, c_in_g, hidden_size, None, kernel_size,
                layers_in_block=1, is_BTC=False, num_layers=n_layers)
        self.post = nn.Conv1d(hidden_size, self.c_half, 1)

    def forward(self, x, nonpadding, cond=None, reverse=False):
        x0, x1 = x[:, :self.c_half], x[:, self.c_half:]
        x_ = self.pre(x0) * nonpadding
        x_ = self.enc(x_, nonpadding=nonpadding, cond=cond)
        m = self.post(x_)
        x1 = m + x1 if not reverse else x1 - m
        x = torch.cat([x0, x1], 1)
        return x * nonpadding


class ResFlow(nn.Module):
    def __init__(self,
                 c_in,
                 hidden_size,
                 kernel_size,
                 n_flow_layers,
                 n_flow_steps=4,
                 c_cond=0,
                 nn_type='wn'):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flow_steps):
            self.flows.append(
                CouplingLayer(c_in, hidden_size, kernel_size, n_flow_layers, c_in_g=c_cond, nn_type=nn_type))
            self.flows.append(FlipLayer())

    def forward(self, x, nonpadding, cond=None, reverse=False):
        for flow in (self.flows if not reverse else reversed(self.flows)):
            x = flow(x, nonpadding, cond=cond, reverse=reverse)
        return x
