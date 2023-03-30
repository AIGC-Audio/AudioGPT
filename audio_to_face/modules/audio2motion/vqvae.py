import scipy
from scipy import linalg
from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
from audio_to_face.modules.audio2motion.transformer_models import FFTBlocks
import audio_to_face.modules.audio2motion.utils as utils
from audio_to_face.modules.audio2motion.flow_base import Glow, WN, ResidualCouplingBlock
import torch.distributions as dist
from audio_to_face.modules.audio2motion.cnn_models import LambdaLayer, LayerNorm

from vector_quantize_pytorch import VectorQuantize


class FVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0, strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.pre_net = nn.Sequential(*[
            nn.Conv1d(in_channels, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            if i == 0 else
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ])
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels * 2, 1)
        self.latent_channels = latent_channels

    def forward(self, x, x_mask, g):
        x = self.pre_net(x)
        x_mask = x_mask[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        m, logs = torch.split(x, self.latent_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs, x_mask


class FVAEDecoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0,
                 strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.pre_net = nn.Sequential(*[
            nn.ConvTranspose1d(latent_channels, hidden_channels, kernel_size=s, stride=s)
            if i == 0 else
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=s, stride=s)
            for i, s in enumerate(strides)
        ])
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask, g):
        x = self.pre_net(x)
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        return x


class VQVAE(nn.Module):
    def __init__(self,
                 in_out_channels=64, hidden_channels=256, latent_size=16,
                 kernel_size=3, enc_n_layers=5, dec_n_layers=5, gin_channels=80, strides=[4,],
                 sqz_prior=False):
        super().__init__()
        self.in_out_channels = in_out_channels
        self.strides = strides
        self.hidden_size = hidden_channels
        self.latent_size = latent_size
        self.g_pre_net = nn.Sequential(*[
            nn.Conv1d(gin_channels, gin_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ])
        self.encoder = FVAEEncoder(in_out_channels, hidden_channels, hidden_channels, kernel_size,
                                   enc_n_layers, gin_channels, strides=strides)
        # if use_prior_glow:
        #     self.prior_flow = ResidualCouplingBlock(
        #         latent_size, glow_hidden, glow_kernel_size, 1, glow_n_blocks, 4, gin_channels=gin_channels)
        self.vq = VectorQuantize(dim=hidden_channels, codebook_size=256, codebook_dim=16)

        self.decoder = FVAEDecoder(hidden_channels, hidden_channels, in_out_channels, kernel_size,
                                   dec_n_layers, gin_channels, strides=strides)
        self.prior_dist = dist.Normal(0, 1)
        self.sqz_prior = sqz_prior

    def forward(self, x=None, x_mask=None, g=None, infer=False, **kwargs):
        """

        :param x: [B, T,  C_in_out]
        :param x_mask: [B, T]
        :param g: [B, T, C_g]
        :return:
        """
        x_mask = x_mask[:, None, :] # [B, 1, T]
        g = g.transpose(1,2) # [B, C_g, T]
        g_for_sqz = g

        g_sqz = self.g_pre_net(g_for_sqz)

        if not infer:
            x = x.transpose(1,2) # [B, C, T]
            z_q, m_q, logs_q, x_mask_sqz = self.encoder(x, x_mask, g_sqz)
            if self.sqz_prior:
                z_q = F.interpolate(z_q, scale_factor=1/8)
            z_p, idx, commit_loss = self.vq(z_q.transpose(1,2))
            if self.sqz_prior:
                z_p = F.interpolate(z_p.transpose(1,2),scale_factor=8).transpose(1,2)

            x_recon = self.decoder(z_p.transpose(1,2), x_mask, g)
            return x_recon.transpose(1,2), commit_loss, z_p.transpose(1,2), m_q.transpose(1,2), logs_q.transpose(1,2)
        else:
            bs, t = g_sqz.shape[0], g_sqz.shape[2]
            if self.sqz_prior:
                t = t // 8
            latent_shape = [int(bs * t)]
            latent_idx = torch.randint(0,256,latent_shape).to(self.vq.codebook.device)
            # latent_idx = torch.ones_like(latent_idx, dtype=torch.long)
            # z_p = torch.gather(self.vq.codebook, 0, latent_idx)# self.vq.codebook[latent_idx]
            z_p = self.vq.codebook[latent_idx]
            z_p = z_p.reshape([bs, t, -1])
            z_p = self.vq.project_out(z_p)
            if self.sqz_prior:
                z_p = F.interpolate(z_p.transpose(1,2),scale_factor=8).transpose(1,2)

            x_recon = self.decoder(z_p.transpose(1,2), 1, g)
            return x_recon.transpose(1,2), z_p.transpose(1,2)


class VQVAEModel(nn.Module):
    def __init__(self, in_out_dim=71, sqz_prior=False, enc_no_cond=False):
        super().__init__()
        self.mel_encoder = nn.Sequential(*[
                nn.Conv1d(80, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, 3, 1, 1, bias=False)
            ]) 
        self.in_dim, self.out_dim = in_out_dim, in_out_dim
        self.sqz_prior = sqz_prior
        self.enc_no_cond = enc_no_cond
        self.vae = VQVAE(in_out_channels=in_out_dim, hidden_channels=256, latent_size=16, kernel_size=5,
            enc_n_layers=8, dec_n_layers=4, gin_channels=64, strides=[4,], sqz_prior=sqz_prior)
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x.transpose(1,2), scale_factor=0.5, mode='nearest').transpose(1,2))
    
    @property
    def device(self):
        return self.vae.parameters().__next__().device

    def forward(self, batch, ret, log_dict=None, train=True):
        infer = not train
        mask = batch['y_mask'].to(self.device)
        mel = batch['mel'].to(self.device)
        mel = self.downsampler(mel)

        mel_feat = self.mel_encoder(mel.transpose(1,2)).transpose(1,2)
        if not infer:
            exp = batch['exp'].to(self.device)
            pose = batch['pose'].to(self.device)
            if self.in_dim == 71:
                x = torch.cat([exp, pose], dim=-1) # [B, T, C=64 + 7]
            elif self.in_dim == 64:
                x = exp
            elif self.in_dim == 7:
                x = pose
            if self.enc_no_cond:
                x_recon, loss_commit, z_p, m_q, logs_q = self.vae(x=x, x_mask=mask, g=torch.zeros_like(mel_feat), infer=False)
            else:
                x_recon, loss_commit, z_p, m_q, logs_q = self.vae(x=x, x_mask=mask, g=mel_feat, infer=False)
            loss_commit = loss_commit.reshape([])
            ret['pred'] = x_recon
            ret['mask'] = mask
            ret['loss_commit'] = loss_commit
            return x_recon, loss_commit, m_q, logs_q
        else:
            x_recon, z_p = self.vae(x=None, x_mask=mask, g=mel_feat, infer=True)
            return x_recon

    # def __get_feat(self, exp, pose):
    # diff_exp = exp[:-1, :] - exp[1:, :]
    # exp_std = (np.std(exp, axis = 0) - self.exp_std_mean) / self.exp_std_std
    # diff_exp_std = (np.std(diff_exp, axis = 0) - self.exp_diff_std_mean) / self.exp_diff_std_std

    # diff_pose = pose[:-1, :] - pose[1:, :]
    # diff_pose_std = (np.std(diff_pose, axis = 0) - self.pose_diff_std_mean) / self.pose_diff_std_std

    # return np.concatenate((exp_std, diff_exp_std, diff_pose_std))
    
    def num_params(self, model, print_out=True, model_name="model"):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters
