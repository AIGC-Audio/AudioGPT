import librosa
import numpy as np
import torch
import torch.nn.functional as F


def _stft(y, hop_size, win_size, fft_size):
    return librosa.stft(y=y, n_fft=fft_size, hop_length=hop_size, win_length=win_size, pad_mode='constant')


def _istft(y, hop_size, win_size):
    return librosa.istft(y, hop_length=hop_size, win_length=win_size)


def griffin_lim(S, hop_size, win_size, fft_size, angles=None, n_iters=30):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape)) if angles is None else angles
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hop_size, win_size)
    for i in range(n_iters):
        angles = np.exp(1j * np.angle(_stft(y, hop_size, win_size, fft_size)))
        y = _istft(S_complex * angles, hop_size, win_size)
    return y


def istft(amp, ang, hop_size, win_size, fft_size, pad=False, window=None):
    spec = amp * torch.exp(1j * ang)
    spec_r = spec.real
    spec_i = spec.imag
    spec = torch.stack([spec_r, spec_i], -1)
    if window is None:
        window = torch.hann_window(win_size).to(amp.device)
    if pad:
        spec = F.pad(spec, [0, 0, 0, 1], mode='reflect')
    wav = torch.istft(spec, fft_size, hop_size, win_size)
    return wav


def griffin_lim_torch(S, hop_size, win_size, fft_size, angles=None, n_iters=30):
    """

    Examples:
    >>> x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size, win_length=win_length, pad_mode="constant")
    >>> x_stft = x_stft[None, ...]
    >>> amp = np.abs(x_stft)
    >>> angle_init = np.exp(2j * np.pi * np.random.rand(*x_stft.shape))
    >>> amp = torch.FloatTensor(amp)
    >>> wav = griffin_lim_torch(amp, angle_init, hparams)

    :param amp: [B, n_fft, T]
    :param ang: [B, n_fft, T]
    :return: [B, T_wav]
    """
    angles = torch.exp(2j * np.pi * torch.rand(*S.shape)) if angles is None else angles
    window = torch.hann_window(win_size).to(S.device)
    y = istft(S, angles, hop_size, win_size, fft_size, window=window)
    for i in range(n_iters):
        x_stft = torch.stft(y, fft_size, hop_size, win_size, window)
        x_stft = x_stft[..., 0] + 1j * x_stft[..., 1]
        angles = torch.angle(x_stft)
        y = istft(S, angles, hop_size, win_size, fft_size, window=window)
    return y


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _build_mel_basis(audio_sample_rate, fft_size, audio_num_mel_bins, fmin, fmax):
    assert fmax <= audio_sample_rate // 2
    return librosa.filters.mel(audio_sample_rate, fft_size, n_mels=audio_num_mel_bins, fmin=fmin, fmax=fmax)


def _linear_to_mel(spectogram, audio_sample_rate, fft_size, audio_num_mel_bins, fmin, fmax):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(audio_sample_rate, fft_size, audio_num_mel_bins, fmin, fmax)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, audio_sample_rate, fft_size, audio_num_mel_bins, fmin, fmax):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(audio_sample_rate, fft_size, audio_num_mel_bins, fmin, fmax))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))
