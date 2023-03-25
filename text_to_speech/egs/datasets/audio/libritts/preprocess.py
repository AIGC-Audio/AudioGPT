from data_gen.tts.base_preprocess import BasePreprocessor
import glob, os

class LibriTTSPreprocess(BasePreprocessor):
    def meta_data(self):
        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/*/*/*/*.wav'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            txt_fn = f'{wav_fn[:-4]}.normalized.txt'
            with open(txt_fn, 'r') as f:
                txt = f.read()
            spk_name = item_name.split("_")[0]
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}