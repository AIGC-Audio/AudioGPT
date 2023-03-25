import glob
from data_gen.tts.base_preprocess import BasePreprocessor


class AiShell3Preprocess(BasePreprocessor):
    def meta_data(self):
        wavfn2text = {}

        def get_wavfn2text(dir_name):
            d = open(f'{self.raw_data_dir}/{dir_name}/content.txt').readlines()
            d = [l.strip().split("\t") for l in d if l.strip() != '']
            d = {l[0]: "".join(l[1].split(" ")[::2]) for l in d}
            wavfn2text.update(d)

        get_wavfn2text('train')
        get_wavfn2text('test')

        all_wavs = sorted(
            glob.glob(f'{self.raw_data_dir}/train/wav/*/*.wav') +
            glob.glob(f'{self.raw_data_dir}/test/wav/*/*.wav'))
        for wav_fn in all_wavs:
            wav_basename = wav_fn.split("/")[-1]
            spk_name = wav_fn.split("/")[-2]
            item_name = f'{spk_name}_{wav_basename}'
            # yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': l}
            # yield item_name, wav_fn, wavfn2text[wav_basename], spk_name
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': wavfn2text[wav_basename], 'spk_name': spk_name}


if __name__ == "__main__":
    AiShell3PreAlign().process()
