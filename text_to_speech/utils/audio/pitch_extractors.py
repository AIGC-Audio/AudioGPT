import numpy as np
from text_to_speech.utils.audio.pitch.utils import denorm_f0, norm_f0, f0_to_coarse
import parselmouth

PITCH_EXTRACTOR = {}


def register_pitch_extractor(name):
    def register_pitch_extractor_(cls):
        PITCH_EXTRACTOR[name] = cls
        return cls

    return register_pitch_extractor_


def get_pitch_extractor(name):
    return PITCH_EXTRACTOR[name]


def extract_pitch_simple(wav):
    from text_to_speech.utils.commons.hparams import hparams
    return extract_pitch(hparams['pitch_extractor'], wav,
                         hparams['hop_size'], hparams['audio_sample_rate'],
                         f0_min=hparams['f0_min'], f0_max=hparams['f0_max'])


def extract_pitch(extractor_name, wav_data, hop_size, audio_sample_rate, f0_min=75, f0_max=800, **kwargs):
    return get_pitch_extractor(extractor_name)(wav_data, hop_size, audio_sample_rate, f0_min, f0_max, **kwargs)


@register_pitch_extractor('parselmouth')
def parselmouth_pitch(wav_data, hop_size, audio_sample_rate, f0_min, f0_max,
                      voicing_threshold=0.6, *args, **kwargs):
    import parselmouth
    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(wav_data) // hop_size)
    f0_pm = parselmouth.Sound(wav_data, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode='constant')
    return f0


def get_pitch(wav_data, mel, hparams):
    """
    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
    f0_min = 80
    f0_max = 750

    if hparams['pitch_extractor'] == 'harvest':
        import pyworld as pw
        f0, t = pw.harvest(wav_data.astype(np.double), hparams['audio_sample_rate'],
                           frame_period=hparams['hop_size'] / hparams['audio_sample_rate'] * 1000)
    if hparams['pitch_extractor'] == 'dio':
        _f0, t = pw.dio(wav_data.astype(np.double), hparams['audio_sample_rate'],
                        frame_period=hparams['hop_size'] / hparams['audio_sample_rate'] * 1000)
        f0 = pw.stonemask(wav_data.astype(np.double), _f0, t, hparams['audio_sample_rate'])  # pitch refinement
    elif hparams['pitch_extractor'] == 'parselmouth':
        if hparams['hop_size'] == 128:
            pad_size = 4
        elif hparams['hop_size'] == 256:
            pad_size = 2
        else:
            assert False
        f0 = parselmouth.Sound(wav_data, hparams['audio_sample_rate']).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
        lpad = pad_size * 2
        rpad = len(mel) - len(f0) - lpad
        f0 = np.pad(f0, [[lpad, rpad]], mode='constant')

    # mel和f0是2个库抽的 需要保证两者长度一致
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse