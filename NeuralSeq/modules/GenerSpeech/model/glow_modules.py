import scipy
from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
from modules.commons.common_layers import Permute
from modules.fastspeech.tts_modules import FFTBlocks
from modules.GenerSpeech.model.wavenet import fused_add_tanh_sigmoid_multiply, WN


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



class ActNorm(nn.Module):  # glow中的线性变换层
    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = torch.sum(-self.logs) * x_len
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]
        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):  # 可逆卷积
    def __init__(self, channels, n_split=4, no_jacobian=False, lu=True, n_sqz=2, **kwargs):
        super().__init__()
        assert (n_split % 2 == 0)
        self.channels = channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.no_jacobian = no_jacobian

        w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.lu = lu
        if lu:
            # LU decomposition can slightly speed up the inverse
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_init.shape, dtype=float), -1)
            eye = np.eye(*w_init.shape, dtype=float)

            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))
        else:
            self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert (c % self.n_split == 0)
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, self.n_sqz, c // self.n_split, self.n_split // self.n_sqz, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        if self.lu:
            self.weight, log_s = self._get_weight()
            logdet = log_s.sum()
            logdet = logdet * (c / self.n_split) * x_len
        else:
            logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        if reverse:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = -logdet
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, self.n_sqz, self.n_split // self.n_sqz, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def _get_weight(self):
        l, log_s, u = self.l, self.log_s, self.u
        l = l * self.l_mask + self.eye
        u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
        weight = torch.matmul(self.p, torch.matmul(l, u))
        return weight, log_s

    def store_inverse(self):
        weight, _ = self._get_weight()
        self.weight_inv = torch.inverse(weight.float()).to(next(self.parameters()).device)


class InvConv(nn.Module):
    def __init__(self, channels, no_jacobian=False, lu=True, **kwargs):
        super().__init__()
        w_shape = [channels, channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(float)
        LU_decomposed = lu
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=float), -1)
            eye = np.eye(*w_shape, dtype=float)

            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.weight = None

    def get_weight(self, device, reverse):
        w_shape = self.w_shape
        self.p = self.p.to(device)
        self.sign_s = self.sign_s.to(device)
        self.l_mask = self.l_mask.to(device)
        self.eye = self.eye.to(device)
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        dlogdet = self.log_s.sum()
        if not reverse:
            w = torch.matmul(self.p, torch.matmul(l, u))
        else:
            l = torch.inverse(l.double()).float()
            u = torch.inverse(u.double()).float()
            w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        """
        log-det = log|abs(|W|)| * pixels
        """
        b, c, t = x.size()
        if x_mask is None:
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        logdet = 0
        if not reverse:
            weight, dlogdet = self.get_weight(x.device, reverse)
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet + dlogdet * x_len
            return z, logdet
        else:
            if self.weight is None:
                weight, dlogdet = self.get_weight(x.device, reverse)
            else:
                weight, dlogdet = self.weight, self.dlogdet
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet * x_len
            return z, logdet

    def store_inverse(self):
        self.weight, self.dlogdet = self.get_weight('cuda', reverse=True)


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x, logdet

    def store_inverse(self):
        pass


class CouplingBlock(nn.Module):  # 仿射耦合层
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, p_dropout=0, sigmoid_scale=False,
                 share_cond_layers=False, wn=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        start = torch.nn.utils.weight_norm(start)
        self.start = start
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels,
                     p_dropout, share_cond_layers)
        if wn is not None:
            self.wn.in_layers = wn.in_layers
            self.wn.res_skip_layers = wn.res_skip_layers

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]

        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))
        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = torch.sum(-logs * x_mask, [1, 2])
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


class GlowFFTBlocks(FFTBlocks):
    def __init__(self, hidden_size=128, gin_channels=256, num_layers=2, ffn_kernel_size=5,
                 dropout=None, num_heads=4, use_pos_embed=True, use_last_norm=True,
                 norm='ln', use_pos_embed_alpha=True):
        super().__init__(hidden_size, num_layers, ffn_kernel_size, dropout, num_heads, use_pos_embed,
                         use_last_norm, norm, use_pos_embed_alpha)
        self.inp_proj = nn.Conv1d(hidden_size + gin_channels, hidden_size, 1)

    def forward(self, x, x_mask=None, g=None):
        """
        :param x: [B, C_x, T]
        :param x_mask: [B, 1, T]
        :param g: [B, C_g, T]
        :return: [B, C_x, T]
        """
        if g is not None:
            x = self.inp_proj(torch.cat([x, g], 1))
        x = x.transpose(1, 2)
        x = super(GlowFFTBlocks, self).forward(x, x_mask[:, 0] == 0)
        x = x.transpose(1, 2)
        return x


class TransformerCouplingBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers,
                 gin_channels=0, p_dropout=0, sigmoid_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        self.start = start
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.fft_blocks = GlowFFTBlocks(
            hidden_size=hidden_channels,
            ffn_kernel_size=3,
            gin_channels=gin_channels,
            num_layers=n_layers)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]

        x = self.start(x_0) * x_mask
        x = self.fft_blocks(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))
        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = torch.sum(-logs * x_mask, [1, 2])
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        pass


class FreqFFTCouplingBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers,
                 gin_channels=0, p_dropout=0, sigmoid_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        hs = hidden_channels
        stride = 8
        self.start = torch.nn.Conv2d(3, hs, kernel_size=stride * 2,
                                     stride=stride, padding=stride // 2)
        end = nn.ConvTranspose2d(hs, 2, kernel_size=stride, stride=stride)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = nn.Sequential(
            nn.Conv2d(hs * 3, hs, 3, 1, 1),
            nn.ReLU(),
            nn.GroupNorm(4, hs),
            nn.Conv2d(hs, hs, 3, 1, 1),
            end
        )
        self.fft_v = FFTBlocks(hidden_size=hs, ffn_kernel_size=1, num_layers=n_layers)
        self.fft_h = nn.Sequential(
            nn.Conv1d(hs, hs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(hs, hs, 3, 1, 1),
        )
        self.fft_g = nn.Sequential(
            nn.Conv1d(
                gin_channels - 160, hs, kernel_size=stride * 2, stride=stride, padding=stride // 2),
            Permute(0, 2, 1),
            FFTBlocks(hidden_size=hs, ffn_kernel_size=1, num_layers=n_layers),
            Permute(0, 2, 1),
        )

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        g_, _ = unsqueeze(g)
        g_mel = g_[:, :80]
        g_txt = g_[:, 80:]
        g_mel, _ = squeeze(g_mel)
        g_txt, _ = squeeze(g_txt)  # [B, C, T]

        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]
        x = torch.stack([x_0, g_mel[:, :80], g_mel[:, 80:]], 1)
        x = self.start(x)  # [B, C, N_bins, T]
        B, C, N_bins, T = x.shape

        x_v = self.fft_v(x.permute(0, 3, 2, 1).reshape(B * T, N_bins, C))
        x_v = x_v.reshape(B, T, N_bins, -1).permute(0, 3, 2, 1)
        # x_v = x

        x_h = self.fft_h(x.permute(0, 2, 1, 3).reshape(B * N_bins, C, T))
        x_h = x_h.reshape(B, N_bins, -1, T).permute(0, 2, 1, 3)
        # x_h = x

        x_g = self.fft_g(g_txt)[:, :, None, :].repeat(1, 1, 10, 1)
        x = torch.cat([x_v, x_h, x_g], 1)
        out = self.end(x)

        z_0 = x_0
        m = out[:, 0]
        logs = out[:, 1]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))
        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = torch.sum(-logs * x_mask, [1, 2])
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        pass


class Glow(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_blocks,
                 n_layers,
                 p_dropout=0.,
                 n_split=4,
                 n_sqz=2,
                 sigmoid_scale=False,
                 gin_channels=0,
                 inv_conv_type='near',
                 share_cond_layers=False,
                 share_wn_layers=0,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels
        self.share_cond_layers = share_cond_layers
        if gin_channels != 0 and share_cond_layers:
            cond_layer = torch.nn.Conv1d(gin_channels * n_sqz, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        wn = None
        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            if inv_conv_type == 'near':
                self.flows.append(InvConvNear(channels=in_channels * n_sqz, n_split=n_split, n_sqz=n_sqz))
            if inv_conv_type == 'invconv':
                self.flows.append(InvConv(channels=in_channels * n_sqz))
            if share_wn_layers > 0:
                if b % share_wn_layers == 0:
                    wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels * n_sqz,
                            p_dropout, share_cond_layers)
            self.flows.append(
                CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels * n_sqz,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    share_cond_layers=share_cond_layers,
                    wn=wn
                ))

    def forward(self, x, x_mask=None, g=None, reverse=False, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
        else:
            flows = reversed(self.flows)
        if return_hiddens:
            hs = []
        if self.n_sqz > 1:
            x, x_mask_ = squeeze(x, x_mask, self.n_sqz)
            if g is not None:
                g, _ = squeeze(g, x_mask, self.n_sqz)
            x_mask = x_mask_
        if self.share_cond_layers and g is not None:
            g = self.cond_layer(g)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=reverse)
            if return_hiddens:
                hs.append(x)
            logdet_tot += logdet
        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)
        if return_hiddens:
            return x, logdet_tot, hs
        return x, logdet_tot

    def store_inverse(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
        for f in self.flows:
            f.store_inverse()


class GlowV2(nn.Module):
    def __init__(self,
                 in_channels=256,
                 hidden_channels=256,
                 kernel_size=3,
                 dilation_rate=1,
                 n_blocks=8,
                 n_layers=4,
                 p_dropout=0.,
                 n_split=4,
                 n_split_blocks=3,
                 sigmoid_scale=False,
                 gin_channels=0,
                 share_cond_layers=True):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_split_blocks = n_split_blocks
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.cond_layers = nn.ModuleList()
        self.share_cond_layers = share_cond_layers

        self.flows = nn.ModuleList()
        in_channels = in_channels * 2
        for l in range(n_split_blocks):
            blocks = nn.ModuleList()
            self.flows.append(blocks)
            gin_channels = gin_channels * 2
            if gin_channels != 0 and share_cond_layers:
                cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
                self.cond_layers.append(torch.nn.utils.weight_norm(cond_layer, name='weight'))
            for b in range(n_blocks):
                blocks.append(ActNorm(channels=in_channels))
                blocks.append(InvConvNear(channels=in_channels, n_split=n_split))
                blocks.append(CouplingBlock(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    share_cond_layers=share_cond_layers))

    def forward(self, x=None, x_mask=None, g=None, reverse=False, concat_zs=True,
                noise_scale=0.66, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
            assert x_mask is not None
            zs = []
            if return_hiddens:
                hs = []
            for i, blocks in enumerate(flows):
                x, x_mask = squeeze(x, x_mask)
                g_ = None
                if g is not None:
                    g, _ = squeeze(g)
                    if self.share_cond_layers:
                        g_ = self.cond_layers[i](g)
                    else:
                        g_ = g
                for layer in blocks:
                    x, logdet = layer(x, x_mask=x_mask, g=g_, reverse=reverse)
                    if return_hiddens:
                        hs.append(x)
                    logdet_tot += logdet
                if i == self.n_split_blocks - 1:
                    zs.append(x)
                else:
                    x, z = torch.chunk(x, 2, 1)
                    zs.append(z)
            if concat_zs:
                zs = [z.reshape(x.shape[0], -1) for z in zs]
                zs = torch.cat(zs, 1)  # [B, C*T]
            if return_hiddens:
                return zs, logdet_tot, hs
            return zs, logdet_tot
        else:
            flows = reversed(self.flows)
            if x is not None:
                assert isinstance(x, list)
                zs = x
            else:
                B, _, T = g.shape
                zs = self.get_prior(B, T, g.device, noise_scale)
            zs_ori = zs
            if g is not None:
                g_, g = g, []
                for i in range(len(self.flows)):
                    g_, _ = squeeze(g_)
                    g.append(self.cond_layers[i](g_) if self.share_cond_layers else g_)
            else:
                g = [None for _ in range(len(self.flows))]
            if x_mask is not None:
                x_masks = []
                for i in range(len(self.flows)):
                    x_mask, _ = squeeze(x_mask)
                    x_masks.append(x_mask)
            else:
                x_masks = [None for _ in range(len(self.flows))]
            x_masks = x_masks[::-1]
            g = g[::-1]
            zs = zs[::-1]
            x = None
            for i, blocks in enumerate(flows):
                x = zs[i] if x is None else torch.cat([x, zs[i]], 1)
                for layer in reversed(blocks):
                    x, logdet = layer(x, x_masks=x_masks[i], g=g[i], reverse=reverse)
                    logdet_tot += logdet
                x, _ = unsqueeze(x)
            return x, logdet_tot, zs_ori

    def store_inverse(self):
        for f in self.modules():
            if hasattr(f, 'store_inverse') and f != self:
                f.store_inverse()

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

    def get_prior(self, B, T, device, noise_scale=0.66):
        C = 80
        zs = []
        for i in range(len(self.flows)):
            C, T = C, T // 2
            if i == self.n_split_blocks - 1:
                zs.append(torch.randn(B, C * 2, T).to(device) * noise_scale)
            else:
                zs.append(torch.randn(B, C, T).to(device) * noise_scale)
        return zs


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask
