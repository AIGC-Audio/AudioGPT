import os

from data_gen.tts.base_preprocess import BasePreprocessor
import glob


class LibrittsPreAlign(BasePreprocessor):
    def meta_data(self):
        wav_fns = sorted(glob.glob(f'{self.raw_data_dir}/*/*/*.wav'))
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            txt_fn = f'{wav_fn[:-4]}.normalized.txt'
            with open(txt_fn, 'r') as f:
                txt = f.readlines()
                f.close()
            spk = item_name.split("_")[0]
            # Example:
            #
            # 'item_name': '103_1241_000000_000001'
            # 'wav_fn': 'LibriTTS/train-clean-100/103/1241/103_1241_000000_000001.wav'
            # 'txt': 'matthew Cuthbert is surprised'
            # 'spk_name': '103'
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt[0], 'spk_name': spk}


if __name__ == "__main__":
    LibrittsPreAlign().process()
