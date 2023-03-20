import subprocess
import matplotlib

matplotlib.use('Agg')
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav *= 32767
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def get_hop_size(hparams):
    hop_size = hparams['hop_size']
    if hop_size is None:
        assert hparams['frame_shift_ms'] is not None
        hop_size = int(hparams['frame_shift_ms'] / 1000 * hparams['audio_sample_rate'])
    return hop_size


###########################################################################################
def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams['fft_size'], hop_length=get_hop_size(hparams),
                        win_length=hparams['win_size'], pad_mode='constant')


def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams['win_size'])


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def normalize(S, hparams):
    return (S - hparams['min_level_db']) / -hparams['min_level_db']
