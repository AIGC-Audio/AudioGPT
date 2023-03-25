import numpy as np
import torch


def to_lf0(f0):
    f0[f0 < 1.0e-5] = 1.0e-6
    lf0 = f0.log() if isinstance(f0, torch.Tensor) else np.log(f0)
    lf0[f0 < 1.0e-5] = - 1.0E+10
    return lf0


def to_f0(lf0):
    f0 = np.where(lf0 <= 0, 0.0, np.exp(lf0))
    return f0.flatten()


def f0_to_coarse(f0, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(), f0.min(), f0.max())
    return f0_coarse


def coarse_to_f0(f0_coarse, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    uv = f0_coarse == 1
    f0 = f0_mel_min + (f0_coarse - 1) * (f0_mel_max - f0_mel_min) / (f0_bin - 2)
    f0 = ((f0 / 1127).exp() - 1) * 700
    f0[uv] = 0
    return f0


def norm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == 'log':
        f0 = torch.log2(f0 + 1e-8) if is_torch else np.log2(f0 + 1e-8)
    if uv is not None:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0, pitch_norm='log', f0_mean=None, f0_std=None):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, pitch_norm, f0_mean, f0_std)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    if is_torch:
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        f0 = f0.to(device)
        uv = uv.to(device)
    return f0, uv


def denorm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100, pitch_padding=None, min=50, max=900):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == 'log':
        f0 = 2 ** f0
    f0 = f0.clamp(min=min, max=max) if is_torch else np.clip(f0, a_min=min, a_max=max)
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0
