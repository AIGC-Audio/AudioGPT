import glob
from data_gen.tts.base_preprocess import BasePreprocessor


class WenetSpeechPreprocess(BasePreprocessor):
    def meta_data(self):
        wavfn2text = {}

        def get_wavfn2text():
            d = open(f'{self.raw_data_dir}/extracted_wav/wenetspeech.txt').readlines()
            d = [l.strip().split("\t") for l in d if l.strip() != '' and 'podcast' in l]
            d = {l[0]: l[1] for l in d}
            wavfn2text.update(d)

        get_wavfn2text()

        all_wavs = sorted(wavfn2text.keys())

        for wav_fn in all_wavs:
            wav_basename = wav_fn.split("/")[-2]+"_"+wav_fn.split("/")[-1]
            spk_name = 'asr_data'
            item_name = f'{spk_name}_{wav_basename}'
            yield {
                'item_name': item_name, 
                'wav_fn': wav_fn.replace("/home/jzy/dict_idea/NeuralSeq/", ""), 
                'txt': wavfn2text[wav_fn], 
                'spk_name': spk_name
                }



if __name__ == "__main__":
    WenetSpeechPreprocess.process()
