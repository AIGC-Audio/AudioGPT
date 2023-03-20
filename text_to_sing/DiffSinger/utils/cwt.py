import librosa
import numpy as np
from pycwt import wavelet
from scipy.interpolate import interp1d


def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav


def convert_continuos_f0(f0):
    '''CONVERT F0 TO CONTINUOUS F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    '''
    # get uv information as binary
    f0 = np.copy(f0)
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        print("| all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def get_cont_lf0(f0, frame_period=5.0):
    uv, cont_f0_lpf = convert_continuos_f0(f0)
    # cont_f0_lpf = low_pass_filter(cont_f0_lpf, int(1.0 / (frame_period * 0.001)), cutoff=20)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf


def get_lf0_cwt(lf0):
    '''
    input:
        signal of shape (N)
    output:
        Wavelet_lf0 of shape(10, N), scales of shape(10)
    '''
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    J = 9

    Wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    # Wavelet.shape => (J + 1, len(lf0))
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales


def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = Wavelet_lf0.mean(0)[None, :]
    std = Wavelet_lf0.std(0)[None, :]
    Wavelet_lf0_norm = (Wavelet_lf0 - mean) / std
    return Wavelet_lf0_norm, mean, std


def normalize_cwt_lf0(f0, mean, std):
    uv, cont_lf0_lpf = get_cont_lf0(f0)
    cont_lf0_norm = (cont_lf0_lpf - mean) / std
    Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_norm)
    Wavelet_lf0_norm, _, _ = norm_scale(Wavelet_lf0)

    return Wavelet_lf0_norm


def get_lf0_cwt_norm(f0s, mean, std):
    uvs = list()
    cont_lf0_lpfs = list()
    cont_lf0_lpf_norms = list()
    Wavelet_lf0s = list()
    Wavelet_lf0s_norm = list()
    scaless = list()

    means = list()
    stds = list()
    for f0 in f0s:
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std

        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)  # [560,10]
        Wavelet_lf0_norm, mean_scale, std_scale = norm_scale(Wavelet_lf0)  # [560,10],[1,10],[1,10]

        Wavelet_lf0s_norm.append(Wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        Wavelet_lf0s.append(Wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return Wavelet_lf0s_norm, scaless, means, stds


def inverse_cwt_torch(Wavelet_lf0, scales):
    import torch
    b = ((torch.arange(0, len(scales)).float().to(Wavelet_lf0.device)[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
    return lf0_rec_sum


def inverse_cwt(Wavelet_lf0, scales):
    b = ((np.arange(0, len(scales))[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdims=True)) / lf0_rec_sum.std(-1, keepdims=True)
    return lf0_rec_sum


def cwt2f0(cwt_spec, mean, std, cwt_scales):
    assert len(mean.shape) == 1 and len(std.shape) == 1 and len(cwt_spec.shape) == 3
    import torch
    if isinstance(cwt_spec, torch.Tensor):
        f0 = inverse_cwt_torch(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = f0.exp()  # [B, T]
    else:
        f0 = inverse_cwt(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = np.exp(f0)  # [B, T]
    return f0
