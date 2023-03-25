from data_gen.tts.base_preprocess import BasePreprocessor
import glob, os

class GigaSpeechPreprocess(BasePreprocessor):
    def meta_data(self):
        lj_raw_data_dir = 'data/raw/LJSpeech-1.1'
        for l in list(open(f'{lj_raw_data_dir}/metadata.csv').readlines())[600:]:
            item_name, _, txt = l.strip().split("|")
            wav_fn = f"{lj_raw_data_dir}/wavs/{item_name}.wav"
            txt = txt.lower()
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': 'LJSPK'}

        dirs = sorted(glob.glob(f'{self.raw_data_dir}/*/*/*'))
        for d in dirs:
            txt_fn = glob.glob(f'{d}/*.txt')[0]
            with open(txt_fn, 'r') as f:
                item_name2txt = [l.strip().split(" ") for l in f.readlines()]
            item_name2txt = {x[0]: ' '.join(x[1:]) for x in item_name2txt}
            wav_fns = sorted(glob.glob(f'{d}/*.flac'))
            for wav_fn in wav_fns:
                item_name = os.path.basename(wav_fn)[:-5]
                txt = item_name2txt[item_name].lower()
                spk = item_name.split("-")[0]
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk}
                