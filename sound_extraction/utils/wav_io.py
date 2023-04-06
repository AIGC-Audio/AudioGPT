import librosa
import librosa.filters
import math
import numpy as np
import scipy.io.wavfile

def load_wav(path):
    max_length = 32000 * 10
    wav = librosa.core.load(path, sr=32000)[0]
    if len(wav) > max_length:
        audio = wav[0:max_length]

    # pad audio to max length, 10s for AudioCaps
    if len(wav) < max_length:
        # audio = torch.nn.functional.pad(audio, (0, self.max_length - audio.size(1)), 'constant')
        wav = np.pad(wav, (0, max_length - len(wav)), 'constant')
    wav = wav[...,None]
    return wav


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(path, 32000, wav.astype(np.int16))