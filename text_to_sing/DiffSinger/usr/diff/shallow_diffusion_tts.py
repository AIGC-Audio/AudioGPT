import math
import random
from collections import deque
from functools import partial
from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange

from modules.fastspeech.fs2 import FastSpeech2
from modules.diffsinger_midi.fs2 import FastSpeech2MIDI
from utils.hparams import hparams



def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class GaussianDiffusion(nn.Module):
    def __init__(self, phone_encoder, out_dims, denoise_fn,
                 timesteps=1000, K_step=1000, loss_type=hparams.get('diff_loss_type', 'l1'), betas=None, spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        if hparams.get('use_midi') is not None and hparams['use_midi']:
            self.fs2 = FastSpeech2MIDI(phone_encoder, out_dims)
        else:
            self.fs2 = FastSpeech2(phone_encoder, out_dims)
        self.mel_bins = out_dims

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            if 'schedule_type' in hparams.keys():
                betas = beta_schedule[hparams['schedule_type']](timesteps)
            else:
                betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            if t[0] < interval:
                a_prev = torch.ones_like(a_t)
            else:
                a_prev = extract(self.alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t-interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        elif len(noise_list) >= 3:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)

        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(1)).mean()
            else:
                # print('are you sure w/o nonpadding?')
                loss = (noise - x_recon).abs().mean()

        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=(not infer), infer=infer, **kwargs)
        cond = ret['decoder_inp'].transpose(1, 2)

        if not infer:
            t = torch.randint(0, self.K_step, (b,), device=device).long()
            x = ref_mels
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            ret['diff_loss'] = self.p_losses(x, t, cond)
            # nonpadding = (mel2ph != 0).float()
            # ret['diff_loss'] = self.p_losses(x, t, cond, nonpadding=nonpadding)
        else:
            ret['fs2_mel'] = ret['mel_out']
            fs2_mels = ret['mel_out']
            t = self.K_step
            fs2_mels = self.norm_spec(fs2_mels)
            fs2_mels = fs2_mels.transpose(1, 2)[:, None, :, :]

            x = self.q_sample(x_start=fs2_mels, t=torch.tensor([t - 1], device=device).long())
            if hparams.get('gaussian_start') is not None and hparams['gaussian_start']:
                print('===> gaussion start.')
                shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
                x = torch.randn(shape, device=device)

            if hparams.get('pndm_speedup'):
                print('===> pndm speed:', hparams['pndm_speedup'])
                self.noise_list = deque(maxlen=4)
                iteration_interval = hparams['pndm_speedup']
                for i in tqdm(reversed(range(0, t, iteration_interval)), desc='sample time step',
                              total=t // iteration_interval):
                    x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), iteration_interval,
                                           cond)
            else:
                for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x[:, 0].transpose(1, 2)
            if mel2ph is not None:  # for singing
                ret['mel_out'] = self.denorm_spec(x) * ((mel2ph > 0).float()[:, :, None])
            else:
                ret['mel_out'] = self.denorm_spec(x)
        return ret

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x


class OfflineGaussianDiffusion(GaussianDiffusion):
    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        b, *_, device = *txt_tokens.shape, txt_tokens.device

        ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=True, infer=True, **kwargs)
        cond = ret['decoder_inp'].transpose(1, 2)
        fs2_mels = ref_mels[1]
        ref_mels = ref_mels[0]

        if not infer:
            t = torch.randint(0, self.K_step, (b,), device=device).long()
            x = ref_mels
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            ret['diff_loss'] = self.p_losses(x, t, cond)
        else:
            t = self.K_step
            fs2_mels = self.norm_spec(fs2_mels)
            fs2_mels = fs2_mels.transpose(1, 2)[:, None, :, :]

            x = self.q_sample(x_start=fs2_mels, t=torch.tensor([t - 1], device=device).long())

            if hparams.get('gaussian_start') is not None and hparams['gaussian_start']:
                print('===> gaussion start.')
                shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
                x = torch.randn(shape, device=device)
            for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)
        return ret
