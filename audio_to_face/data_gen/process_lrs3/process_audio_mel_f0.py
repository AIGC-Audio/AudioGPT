import numpy as np
import torch
import glob
import os
import tqdm
import librosa
import parselmouth
from audio_to_face.utils.commons.pitch_utils import f0_to_coarse
from audio_to_face.utils.commons.multiprocess_utils import multiprocess_run_tqdm


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

def extract_mel_from_fname(wav_path,
                      fft_size=512,
                      hop_size=320,
                      win_length=512,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=16000,
                      min_level_db=-100):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=False)
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    mel = mel.T

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)

    return wav.T, mel

def extract_f0_from_wav_and_mel(wav, mel,
                        hop_size=320,
                        audio_sample_rate=16000,
                        ):
    time_step = hop_size / audio_sample_rate * 1000
    f0_min = 80
    f0_max = 750
    f0 = parselmouth.Sound(wav, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse

def extract_mel_f0_from_fname(fname, out_name=None):
    assert fname.endswith(".wav")
    if out_name is None:
        out_name = fname[:-4] + '_audio.npy'

    wav, mel = extract_mel_from_fname(fname)
    f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
    out_dict = {
        "mel": mel, # [T, 80]
        "f0": f0,
    }
    np.save(out_name, out_dict)
    return True

if __name__ == '__main__':
    import os, glob
    lrs3_dir = "/home/yezhenhui/datasets/raw/lrs3_raw"
    wav_name_pattern = os.path.join(lrs3_dir, "*/*.wav")
    wav_names = glob.glob(wav_name_pattern)
    wav_names = sorted(wav_names)
    for _ in multiprocess_run_tqdm(extract_mel_f0_from_fname, args=wav_names, num_workers=32,desc='extracting Mel and f0'):
        pass