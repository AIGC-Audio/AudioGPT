import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
import numpy as np

from audio_to_face.modules.audio2motion.flow_base import Glow, WN, ResidualCouplingBlock
from audio_to_face.modules.audio2motion.transformer_base import Embedding

from audio_to_face.utils.commons.pitch_utils import f0_to_coarse


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e4)  # an arbitrary large number

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

class FVAE(nn.Module):
    def __init__(self,
                 in_out_channels=64, hidden_channels=256, latent_size=16,
                 kernel_size=3, enc_n_layers=5, dec_n_layers=5, gin_channels=80, strides=[4,],
                 use_prior_glow=True, glow_hidden=256, glow_kernel_size=3, glow_n_blocks=5,
                 sqz_prior=False, use_pos_emb=False):
        super(FVAE, self).__init__()
        self.in_out_channels = in_out_channels
        self.strides = strides
        self.hidden_size = hidden_channels
        self.latent_size = latent_size
        self.use_prior_glow = use_prior_glow
        self.sqz_prior = sqz_prior
        self.g_pre_net = nn.Sequential(*[
            nn.Conv1d(gin_channels, gin_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ])
        self.encoder = FVAEEncoder(in_out_channels, hidden_channels, latent_size, kernel_size,
                                   enc_n_layers, gin_channels, strides=strides)
        if use_prior_glow:
            self.prior_flow = ResidualCouplingBlock(
                latent_size, glow_hidden, glow_kernel_size, 1, glow_n_blocks, 4, gin_channels=gin_channels)
        self.use_pos_embed = use_pos_emb
        if sqz_prior:
            self.query_proj = nn.Linear(latent_size, latent_size)
            self.key_proj = nn.Linear(latent_size, latent_size)
            self.value_proj = nn.Linear(latent_size, hidden_channels)
            if self.in_out_channels in [7, 64]:
                self.decoder = FVAEDecoder(hidden_channels, hidden_channels, in_out_channels, kernel_size,
                                    dec_n_layers, gin_channels, strides=strides)
            elif self.in_out_channels == 71:
                self.exp_decoder = FVAEDecoder(hidden_channels, hidden_channels, 64, kernel_size,
                                    dec_n_layers, gin_channels, strides=strides)
                self.pose_decoder = FVAEDecoder(hidden_channels, hidden_channels, 7, kernel_size,
                                    dec_n_layers, gin_channels, strides=strides)
            if self.use_pos_embed:
                self.embed_positions = SinusoidalPositionalEmbedding(self.latent_size, 0,init_size=2000+1,)
        else:
            self.decoder = FVAEDecoder(latent_size, hidden_channels, in_out_channels, kernel_size,
                                    dec_n_layers, gin_channels, strides=strides)

        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, x_mask=None, g=None, infer=False, temperature=1. , **kwargs):
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
                z = z_q
                if self.use_pos_embed:
                    position = self.embed_positions(z.transpose(1,2).abs().sum(-1)).transpose(1,2)
                    z = z + position
                q = self.query_proj(z.mean(dim=-1,keepdim=True).transpose(1,2)) # [B, 1, C=16]
                k = self.key_proj(z.transpose(1,2)) # [B, T, C=16]
                v = self.value_proj(z.transpose(1,2)) # [B, T, C=256]
                attn = torch.bmm(q,k.transpose(1,2)) # [B, 1, T]
                attn = F.softmax(attn, dim=-1)
                out = torch.bmm(attn, v) # [B, 1, C=256]
                style_encoding = out.repeat([1,z_q.shape[-1],1]).transpose(1,2) # [B, C=256, T]
                if self.in_out_channels == 71:
                    x_recon = torch.cat([self.exp_decoder(style_encoding, x_mask, g), self.pose_decoder(style_encoding, x_mask, g)], dim=1)
                else:
                    x_recon = self.decoder(style_encoding, x_mask, g)
            else:
                if self.in_out_channels == 71:
                    x_recon = torch.cat([self.exp_decoder(z_q, x_mask, g), self.pose_decoder(z_q, x_mask, g)], dim=1)
                else:
                    x_recon = self.decoder(z_q, x_mask, g)
            q_dist = dist.Normal(m_q, logs_q.exp())
            if self.use_prior_glow:
                logqx = q_dist.log_prob(z_q)
                z_p = self.prior_flow(z_q, x_mask_sqz, g_sqz)
                logpx = self.prior_dist.log_prob(z_p)
                loss_kl = ((logqx - logpx) * x_mask_sqz).sum() / x_mask_sqz.sum() / logqx.shape[1]
            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
                loss_kl = (loss_kl * x_mask_sqz).sum() / x_mask_sqz.sum() / z_q.shape[1]
                z_p = z_q
            return x_recon.transpose(1,2), loss_kl, z_p.transpose(1,2), m_q.transpose(1,2), logs_q.transpose(1,2)
        else:
            latent_shape = [g_sqz.shape[0], self.latent_size, g_sqz.shape[2]]
            z_p = self.prior_dist.sample(latent_shape).to(g.device) * temperature # [B, latent_size, T_sqz]
            if self.use_prior_glow:
                z_p = self.prior_flow(z_p, 1, g_sqz, reverse=True)
            if self.sqz_prior:
                z = z_p
                if self.use_pos_embed:
                    position = self.embed_positions(z.abs().sum(-1))
                    z += position
                q = self.query_proj(z.mean(dim=-1,keepdim=True).transpose(1,2)) # [B, 1, C=16]
                k = self.key_proj(z.transpose(1,2)) # [B, T, C=16]
                v = self.value_proj(z.transpose(1,2)) # [B, T, C=256]
                attn = torch.bmm(q,k.transpose(1,2)) # [B, 1, T]
                attn = F.softmax(attn, dim=-1)
                out = torch.bmm(attn, v) # [B, 1, C=256]
                style_encoding = out.repeat([1,z_p.shape[-1],1]).transpose(1,2) # [B, C=256, T]
                x_recon = self.decoder(style_encoding, 1, g)
                if self.in_out_channels == 71:
                    x_recon = torch.cat([self.exp_decoder(style_encoding, 1, g), self.pose_decoder(style_encoding, 1, g)], dim=1)
                else:
                    x_recon = self.decoder(style_encoding, 1, g)
            else:
                if self.in_out_channels == 71:
                    x_recon = torch.cat([self.exp_decoder(z_p, 1, g), self.pose_decoder(z_p, 1, g)], dim=1)
                else:
                    x_recon = self.decoder(z_p, 1, g)
            return x_recon.transpose(1,2), z_p.transpose(1,2)


class VAEModel(nn.Module):
    def __init__(self, in_out_dim=64, sqz_prior=False, cond_drop=False, use_prior_flow=True):
        super().__init__()
        mel_feat_dim = 64
        mel_in_dim = 1024 # hubert

        cond_dim = mel_feat_dim
        self.mel_encoder = nn.Sequential(*[
                nn.Conv1d(mel_in_dim, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, mel_feat_dim, 3, 1, 1, bias=False)
            ]) 
        self.cond_drop = cond_drop
        if self.cond_drop:
            self.dropout = nn.Dropout(0.5)

        self.in_dim, self.out_dim = in_out_dim, in_out_dim
        self.sqz_prior = sqz_prior
        self.use_prior_flow = use_prior_flow
        self.vae = FVAE(in_out_channels=in_out_dim, hidden_channels=256, latent_size=16, kernel_size=5,
            enc_n_layers=8, dec_n_layers=4, gin_channels=cond_dim, strides=[4,],
            use_prior_glow=self.use_prior_flow, glow_hidden=64, glow_kernel_size=3, glow_n_blocks=4,sqz_prior=sqz_prior)
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x.transpose(1,2), scale_factor=0.5, mode='nearest').transpose(1,2))

    def num_params(self, model, print_out=True, model_name="model"):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters
    
    @property
    def device(self):
        return self.vae.parameters().__next__().device

    def forward(self, batch, ret, train=True, return_latent=False, temperature=1.):
        infer = not train
        mask = batch['y_mask'].to(self.device)
        mel = batch['hubert'].to(self.device)
        mel = self.downsampler(mel)
        cond_feat = self.mel_encoder(mel.transpose(1,2)).transpose(1,2)

        if self.cond_drop:
            cond_feat = self.dropout(cond_feat)
        
        if not infer:
            exp = batch['y'].to(self.device)
            x = exp
            x_recon, loss_kl, z_p, m_q, logs_q = self.vae(x=x, x_mask=mask, g=cond_feat, infer=False)
            x_recon = x_recon * mask.unsqueeze(-1)
            ret['pred'] = x_recon
            ret['mask'] = mask
            ret['loss_kl'] = loss_kl
            if return_latent:
                ret['m_q'] = m_q
                ret['z_p'] = z_p
            return x_recon, loss_kl, m_q, logs_q
        else:
            x_recon, z_p = self.vae(x=None, x_mask=mask, g=cond_feat, infer=True, temperature=temperature)
            x_recon = x_recon * mask.unsqueeze(-1)
            ret['pred'] = x_recon
            ret['mask'] = mask

            return x_recon


class PitchContourVAEModel(nn.Module):
    def __init__(self, in_out_dim=64, sqz_prior=False, cond_drop=False, use_prior_flow=True):
        super().__init__()
        mel_feat_dim = 64
        mel_in_dim = 1024 # hubert
        
        cond_dim = mel_feat_dim
        self.mel_encoder = nn.Sequential(*[
                nn.Conv1d(mel_in_dim, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, mel_feat_dim, 3, 1, 1, bias=False)
            ])
        
        self.pitch_embed = Embedding(300, mel_feat_dim, None)
        self.pitch_encoder = nn.Sequential(*[
                nn.Conv1d(mel_feat_dim, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 32, 3, 1, 1, bias=False)
            ])
        cond_dim += 32

        self.cond_drop = cond_drop
        if self.cond_drop:
            self.dropout = nn.Dropout(0.5)

        self.in_dim, self.out_dim = in_out_dim, in_out_dim
        self.sqz_prior = sqz_prior
        self.use_prior_flow = use_prior_flow
        self.vae = FVAE(in_out_channels=in_out_dim, hidden_channels=256, latent_size=16, kernel_size=5,
            enc_n_layers=8, dec_n_layers=4, gin_channels=cond_dim, strides=[4,],
            use_prior_glow=self.use_prior_flow, glow_hidden=64, glow_kernel_size=3, glow_n_blocks=4,sqz_prior=sqz_prior)
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x.transpose(1,2), scale_factor=0.5, mode='nearest').transpose(1,2))

    def num_params(self, model, print_out=True, model_name="model"):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters
    
    @property
    def device(self):
        return self.vae.parameters().__next__().device

    def forward(self, batch, ret, train=True, return_latent=False, temperature=1.):
        infer = not train
        mask = batch['y_mask'].to(self.device)
        mel = batch['hubert'].to(self.device)
        f0 = batch['f0'].to(self.device) # [b,t]
        mel = self.downsampler(mel)
        f0 = self.downsampler(f0.unsqueeze(-1)).squeeze(-1)
        f0_coarse = f0_to_coarse(f0)
        pitch_emb = self.pitch_embed(f0_coarse)
        cond_feat = self.mel_encoder(mel.transpose(1,2)).transpose(1,2)
        pitch_feat = self.pitch_encoder(pitch_emb.transpose(1,2)).transpose(1,2)
        cond_feat = torch.cat([cond_feat, pitch_feat], dim=-1)

        if self.cond_drop:
            cond_feat = self.dropout(cond_feat)
        
        if not infer:
            exp = batch['y'].to(self.device)
            x = exp
            x_recon, loss_kl, z_p, m_q, logs_q = self.vae(x=x, x_mask=mask, g=cond_feat, infer=False)
            x_recon = x_recon * mask.unsqueeze(-1)
            ret['pred'] = x_recon
            ret['mask'] = mask
            ret['loss_kl'] = loss_kl
            if return_latent:
                ret['m_q'] = m_q
                ret['z_p'] = z_p
            return x_recon, loss_kl, m_q, logs_q
        else:
            x_recon, z_p = self.vae(x=None, x_mask=mask, g=cond_feat, infer=True, temperature=temperature)
            x_recon = x_recon * mask.unsqueeze(-1)
            ret['pred'] = x_recon
            ret['mask'] = mask

            return x_recon


if __name__ == '__main__':
    model = FVAE(in_out_channels=64, hidden_channels=128, latent_size=32,kernel_size=3, enc_n_layers=6, dec_n_layers=2, 
        gin_channels=80, strides=[4], use_prior_glow=False, glow_hidden=128, glow_kernel_size=3, glow_n_blocks=3)
    x = torch.rand([8, 64, 1000])
    x_mask = torch.ones([8,1,1000])
    g = torch.rand([8, 80, 1000])
    train_out = model(x,x_mask,g,infer=False)
    x_recon, loss_kl, z_p, m_q, logs_q = train_out
    print(" ")
    infer_out = model(x,x_mask,g,infer=True)
    x_recon, z_p = infer_out
    print(" ")
