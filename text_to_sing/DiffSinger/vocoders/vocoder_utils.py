import librosa

from utils.hparams import hparams
import numpy as np


def denoise(wav, v=0.1):
    spec = librosa.stft(y=wav, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                        win_length=hparams['win_size'], pad_mode='constant')
    spec_m = np.abs(spec)
    spec_m = np.clip(spec_m - v, a_min=0, a_max=None)
    spec_a = np.angle(spec)

    return librosa.istft(spec_m * np.exp(1j * spec_a), hop_length=hparams['hop_size'],
                         win_length=hparams['win_size'])
