import os

from data_gen.tts.base_pre_align import BasePreAlign
import glob


class VCTKPreAlign(BasePreAlign):
    def meta_data(self):
        wav_fns = glob.glob(f'{self.raw_data_dir}/wav48/*/*.wav')
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            spk = item_name.split("_")[0]
            txt_fn = wav_fn.split("/")
            txt_fn[-1] = f'{item_name}.txt'
            txt_fn[-3] = f'txt'
            txt_fn = "/".join(txt_fn)
            if os.path.exists(txt_fn) and os.path.exists(wav_fn):
                yield item_name, wav_fn, (self.load_txt, txt_fn), spk


if __name__ == "__main__":
    VCTKPreAlign().process()
