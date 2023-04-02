import subprocess
import matplotlib
import os
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

def denormalize(D, hparams):
    return (D * -hparams['min_level_db']) + hparams['min_level_db']
def rnnoise(filename, out_fn=None, verbose=False, out_sample_rate=22050):
    assert os.path.exists('./rnnoise/examples/rnnoise_demo'), INSTALL_STR
    if out_fn is None:
        out_fn = f"{filename[:-4]}.denoised.wav"
    out_48k_fn = f"{out_fn}.48000.wav"
    tmp0_fn = f"{out_fn}.0.wav"
    tmp1_fn = f"{out_fn}.1.wav"
    tmp2_fn = f"{out_fn}.2.raw"
    tmp3_fn = f"{out_fn}.3.raw"
    if verbose:
        print("Pre-processing audio...")  # wav to pcm raw
    subprocess.check_call(
        f'sox "{filename}" -G -r48000 "{tmp0_fn}"', shell=True, stdin=subprocess.PIPE)  # convert to raw
    subprocess.check_call(
        f'sox -v 0.95 "{tmp0_fn}" "{tmp1_fn}"', shell=True, stdin=subprocess.PIPE)  # convert to raw
    subprocess.check_call(
        f'ffmpeg -y -i "{tmp1_fn}" -loglevel quiet -f s16le -ac 1 -ar 48000 "{tmp2_fn}"',
        shell=True, stdin=subprocess.PIPE)  # convert to raw
    if verbose:
        print("Applying rnnoise algorithm to audio...")  # rnnoise
    subprocess.check_call(
        f'./rnnoise/examples/rnnoise_demo "{tmp2_fn}" "{tmp3_fn}"', shell=True)

    if verbose:
        print("Post-processing audio...")  # pcm raw to wav
    if filename == out_fn:
        subprocess.check_call(f'rm -f "{out_fn}"', shell=True)
    subprocess.check_call(
        f'sox -t raw -r 48000 -b 16 -e signed-integer -c 1 "{tmp3_fn}" "{out_48k_fn}"', shell=True)
    subprocess.check_call(f'sox "{out_48k_fn}" -G -r{out_sample_rate} "{out_fn}"', shell=True)
    subprocess.check_call(f'rm -f "{tmp0_fn}" "{tmp1_fn}" "{tmp2_fn}" "{tmp3_fn}" "{out_48k_fn}"', shell=True)
    if verbose:
        print("Audio-filtering completed!")