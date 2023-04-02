from modules.commons.common_layers import *
from utils.hparams import hparams
from modules.fastspeech.tts_modules import PitchPredictor
from utils.pitch_utils import denorm_f0


class Prenet(nn.Module):
    def __init__(self, in_dim=80, out_dim=256, kernel=5, n_layers=3, strides=None):
        super(Prenet, self).__init__()
        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=padding, stride=self.strides[l]),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim)
            ))
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """

        :param x: [B, T, 80]
        :return: [L, B, T, H], [B, T, H]
        """
        padding_mask = x.abs().sum(-1).eq(0).data  # [B, T]
        nonpadding_mask_TB = 1 - padding_mask.float()[:, None, :]  # [B, 1, T]
        x = x.transpose(1, 2)
        hiddens = []
        for i, l in enumerate(self.layers):
            nonpadding_mask_TB = nonpadding_mask_TB[:, :, ::self.strides[i]]
            x = l(x) * nonpadding_mask_TB
        hiddens.append(x)
        hiddens = torch.stack(hiddens, 0)  # [L, B, H, T]
        hiddens = hiddens.transpose(2, 3)  # [L, B, T, H]
        x = self.out_proj(x.transpose(1, 2))  # [B, T, H]
        x = x * nonpadding_mask_TB.transpose(1, 2)
        return hiddens, x


class ConvBlock(nn.Module):
    def __init__(self, idim=80, n_chans=256, kernel_size=3, stride=1, norm='gn', dropout=0):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == 'bn':
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == 'in':
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == 'gn':
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        elif self.norm == 'ln':
            self.norm = LayerNorm(n_chans // 16, n_chans)
        elif self.norm == 'wn':
            self.conv = torch.nn.utils.weight_norm(self.conv.conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if not isinstance(self.norm, str):
            if self.norm == 'none':
                pass
            elif self.norm == 'ln':
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvStacks(nn.Module):
    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn',
                 dropout=0, strides=None, res=True):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.res = res
        self.in_proj = Linear(idim, n_chans)
        if strides is None:
            strides = [1] * n_layers
        else:
            assert len(strides) == n_layers
        for idx in range(n_layers):
            self.conv.append(ConvBlock(
                n_chans, n_chans, kernel_size, stride=strides[idx], norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x, return_hiddens=False):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        hiddens = []
        for f in self.conv:
            x_ = f(x)
            x = x + x_ if self.res else x_  # (B, C, Tmax)
            hiddens.append(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)  # (B, Tmax, H)
        if return_hiddens:
            hiddens = torch.stack(hiddens, 1)  # [B, L, C, T]
            return x, hiddens
        return x


class PitchExtractor(nn.Module):
    def __init__(self, n_mel_bins=80, conv_layers=2):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.conv_layers = conv_layers

        self.mel_prenet = Prenet(n_mel_bins, self.hidden_size, strides=[1, 1, 1])
        if self.conv_layers > 0:
            self.mel_encoder = ConvStacks(
                    idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size, n_layers=self.conv_layers)
        self.pitch_predictor = PitchPredictor(
            self.hidden_size, n_chans=self.predictor_hidden,
            n_layers=5, dropout_rate=0.1, odim=2,
            padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])

    def forward(self, mel_input=None):
        ret = {}
        mel_hidden = self.mel_prenet(mel_input)[1]
        if self.conv_layers > 0:
            mel_hidden = self.mel_encoder(mel_hidden)

        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(mel_hidden)

        pitch_padding = mel_input.abs().sum(-1) == 0
        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']

        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            hparams, pitch_padding=pitch_padding)
        return ret