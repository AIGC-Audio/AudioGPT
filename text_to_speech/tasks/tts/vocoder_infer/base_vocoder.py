import librosa
from text_to_speech.utils.audio import librosa_wav2spec
from text_to_speech.utils.commons.hparams import hparams
import numpy as np

REGISTERED_VOCODERS = {}


def register_vocoder(name):
    def _f(cls):
        REGISTERED_VOCODERS[name] = cls
        return cls

    return _f


def get_vocoder_cls(vocoder_name):
    return REGISTERED_VOCODERS.get(vocoder_name)


class BaseVocoder:
    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        wav_spec_dict = librosa_wav2spec(wav_fn, fft_size=hparams['fft_size'],
                                         hop_size=hparams['hop_size'],
                                         win_length=hparams['win_size'],
                                         num_mels=hparams['audio_num_mel_bins'],
                                         fmin=hparams['fmin'],
                                         fmax=hparams['fmax'],
                                         sample_rate=hparams['audio_sample_rate'],
                                         loud_norm=hparams['loud_norm'])
        wav = wav_spec_dict['wav']
        mel = wav_spec_dict['mel']
        return wav, mel

    @staticmethod
    def wav2mfcc(wav_fn):
        fft_size = hparams['fft_size']
        hop_size = hparams['hop_size']
        win_length = hparams['win_size']
        sample_rate = hparams['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13,
                                    n_fft=fft_size, hop_length=hop_size,
                                    win_length=win_length, pad_mode="constant", power=1.0)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta]).T
        return mfcc
