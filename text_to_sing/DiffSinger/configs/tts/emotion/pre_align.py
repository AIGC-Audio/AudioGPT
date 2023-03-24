import os

from data_gen.tts.base_preprocess import BasePreprocessor
import glob
import re

class EmoPreAlign(BasePreprocessor):

    def meta_data(self):
        spks = ['0012', '0011', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
        pattern = re.compile('[\t\n ]+')
        for spk in spks:
            for line in open(f"{self.raw_data_dir}/{spk}/{spk}.txt", 'r'):  # 打开文件
                line = re.sub(pattern, ' ', line)
                if line == ' ': continue
                split_ = line.split(' ')
                txt = ' '.join(split_[1: -2])
                item_name = split_[0]
                emotion = split_[-2]
                wav_fn = f'{self.raw_data_dir}/{spk}/{emotion}/{item_name}.wav'
                yield item_name, wav_fn, txt, spk, emotion


if __name__ == "__main__":
    EmoPreAlign().process()
