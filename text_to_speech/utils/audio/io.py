import subprocess

import numpy as np
from scipy.io import wavfile


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])


def to_mp3(out_path):
    if out_path[-4:] == '.wav':
        out_path = out_path[:-4]
    subprocess.check_call(
        f'ffmpeg -threads 1 -loglevel error -i "{out_path}.wav" -vn -b:a 192k -y -hide_banner -async 1 "{out_path}.mp3"',
        shell=True, stdin=subprocess.PIPE)
    subprocess.check_call(f'rm -f "{out_path}.wav"', shell=True)
