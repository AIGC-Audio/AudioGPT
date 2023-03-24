import os
import subprocess
import librosa
import numpy as np
from data_gen.tts.wav_processors.base_processor import BaseWavProcessor, register_wav_processors
from data_gen.tts.data_gen_utils import trim_long_silences
from utils.audio import save_wav
from utils.hparams import hparams


@register_wav_processors(name='sox_to_wav')
class ConvertToWavProcessor(BaseWavProcessor):
    @property
    def name(self):
        return 'ToWav'

    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        if input_fn[-4:] == '.wav':
            return input_fn, sr
        else:
            output_fn = self.output_fn(input_fn)
            subprocess.check_call(f'sox -v 0.95 "{input_fn}" -t wav "{output_fn}"', shell=True)
            return output_fn, sr


@register_wav_processors(name='sox_resample')
class ResampleProcessor(BaseWavProcessor):
    @property
    def name(self):
        return 'Resample'

    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        output_fn = self.output_fn(input_fn)
        sr_file = librosa.core.get_samplerate(input_fn)
        if sr != sr_file:
            subprocess.check_call(f'sox -v 0.95 "{input_fn}" -r{sr} "{output_fn}"', shell=True)
            y, _ = librosa.core.load(input_fn, sr=sr)
            y, _ = librosa.effects.trim(y)
            save_wav(y, output_fn, sr)
            return output_fn, sr
        else:
            return input_fn, sr


@register_wav_processors(name='trim_sil')
class TrimSILProcessor(BaseWavProcessor):
    @property
    def name(self):
        return 'TrimSIL'

    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        output_fn = self.output_fn(input_fn)
        y, _ = librosa.core.load(input_fn, sr=sr)
        y, _ = librosa.effects.trim(y)
        save_wav(y, output_fn, sr)
        return output_fn


@register_wav_processors(name='trim_all_sil')
class TrimAllSILProcessor(BaseWavProcessor):
    @property
    def name(self):
        return 'TrimSIL'

    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        output_fn = self.output_fn(input_fn)
        y, audio_mask, _ = trim_long_silences(
            input_fn, vad_max_silence_length=preprocess_args.get('vad_max_silence_length', 12))
        save_wav(y, output_fn, sr)
        if preprocess_args['save_sil_mask']:
            os.makedirs(f'{processed_dir}/sil_mask', exist_ok=True)
            np.save(f'{processed_dir}/sil_mask/{item_name}.npy', audio_mask)
        return output_fn, sr

