import librosa
import numpy as np

def get_mel_from_fname(wav_path,
                    fft_size=512,
                    hop_size=320,
                    win_length=512,
                    window="hann",
                    num_mels=80,
                    fmin=80,
                    fmax=7600,
                    eps=1e-6,
                    sample_rate=16000,
                    min_level_db=-100,
                    return_energy=False):
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
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    mel = mel.T
    # f0 = get_pitch(wav, mel)
    if return_energy:
        audio_energy = librosa.feature.rms(y=wav, frame_length=fft_size, hop_length=hop_size, center=False) # 对每一frame计算root-mean-square
        audio_energy = np.transpose(audio_energy) # [t,1]
        return mel, audio_energy
    return  mel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_name', type=str,
                        default='data/processed/videos/FDDM/aud.wav', help='')
    parser.add_argument('--mel_npy_name', type=str,
                        default='data/processed/videos/FDDM/mel.npy', help='')
    args = parser.parse_args()
    mel, energy = get_mel_from_fname(args.wav_name, return_energy=True)
    out_dict = {
        'mel': mel,
        'energy': energy
    }
    np.save(args.mel_npy_name, out_dict)
