from data_gen.tts.base_preprocess import BasePreprocessor
import re


class BiaobeiPreprocess(BasePreprocessor):
    def meta_data(self):
        input_dir = self.raw_data_dir
        with open(f"{input_dir}/ProsodyLabeling/000001-010000.txt", encoding='utf-8') as f:
            bb_lines = f.readlines()[::2]
        for l_idx, l in (enumerate([re.sub("\#\d+", "", l.split('\t')[1].strip()) for l in bb_lines])):
            item_name = f'{l_idx + 1:06d}'
            wav_fn = f"{input_dir}/wav/{l_idx + 1:06d}.wav"
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': l}

if __name__ == "__main__":
    BiaobeiPreprocess().process()
