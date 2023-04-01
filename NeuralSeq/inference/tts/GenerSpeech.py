import torch
import os
import importlib
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.ckpt_utils import load_ckpt, get_last_checkpoint
from modules.GenerSpeech.model.generspeech import GenerSpeech
from data_gen.tts.emotion import inference as EmotionEncoder
from data_gen.tts.emotion.inference import embed_utterance as Embed_utterance
from data_gen.tts.emotion.inference import preprocess_wav
from data_gen.tts.data_gen_utils import is_sil_phoneme
from resemblyzer import VoiceEncoder
from utils import audio
class GenerSpeechInfer(BaseTTSInfer):
    def build_model(self):
        model = GenerSpeech(self.ph_encoder)
        model.eval()
        load_ckpt(model, self.hparams['work_dir'], 'model')
        return model

    def preprocess_input(self, inp):
        """
        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        # processed text
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        ph, txt, word, ph2word, ph_gb_word = preprocessor.txt_to_ph(preprocessor.txt_processor, text_raw, preprocess_args)
        ph_token = self.ph_encoder.encode(ph)

        # processed ref audio
        ref_audio = inp['ref_audio']
        processed_ref_audio = 'example/temp.wav'
        voice_encoder = VoiceEncoder().cuda()
        encoder = [self.ph_encoder, self.word_encoder]
        EmotionEncoder.load_model(self.hparams['emotion_encoder_path'])
        binarizer_cls = self.hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)

        ref_audio_raw, ref_text_raw = self.asr(ref_audio)  # prepare text
        ph_ref, txt_ref, word_ref, ph2word_ref, ph_gb_word_ref = preprocessor.txt_to_ph(preprocessor.txt_processor, ref_text_raw, preprocess_args)
        ph_gb_word_nosil = ["_".join([p for p in w.split("_") if not is_sil_phoneme(p)]) for w in ph_gb_word_ref.split(" ") if not is_sil_phoneme(w)]
        phs_for_align = ['SIL'] + ph_gb_word_nosil + ['SIL']
        phs_for_align = " ".join(phs_for_align)

        # prepare files for alignment
        os.system('rm -r example/; mkdir example/')
        audio.save_wav(ref_audio_raw, processed_ref_audio, self.hparams['audio_sample_rate'])
        with open(f'example/temp.lab', 'w') as f_txt:
            f_txt.write(phs_for_align)
        os.system(f'mfa align example/ {self.hparams["binary_data_dir"]}/mfa_dict.txt {self.hparams["binary_data_dir"]}/mfa_model.zip example/textgrid/  --clean')
        item2tgfn = 'example/textgrid/temp.TextGrid'  # prepare textgrid alignment

        item = binarizer_cls.process_item(item_name, ph_ref, txt_ref, item2tgfn, processed_ref_audio, 0, 0, encoder, self.hparams['binarization_args'])
        item['emo_embed'] = Embed_utterance(preprocess_wav(item['wav_fn']))
        item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])

        item.update({
            'ref_ph': item['ph'],
            'ph': ph,
            'ph_token': ph_token,
            'text': txt
        })
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]

        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        mels = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        f0 = torch.FloatTensor(item['f0'])[None, :].to(self.device)
        # uv = torch.FloatTensor(item['uv']).to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        spk_embed = torch.FloatTensor(item['spk_embed'])[None, :].to(self.device)
        emo_embed = torch.FloatTensor(item['emo_embed'])[None, :].to(self.device)

        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        mel2word = torch.LongTensor(item['mel2word'])[None, :].to(self.device)
        word_tokens = torch.LongTensor(item['word_tokens'])[None, :].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'mels': mels,
            'f0': f0,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_embed': spk_embed,
            'emo_embed': emo_embed,
            'mel2ph': mel2ph,
            'ph2word': ph2word,
            'mel2word': mel2word,
            'word_tokens': word_tokens,
        }
        return batch

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        with torch.no_grad():
            output = self.model(txt_tokens, ref_mel2ph=sample['mel2ph'], ref_mel2word=sample['mel2word'], ref_mels=sample['mels'],
                                spk_embed=sample['spk_embed'], emo_embed=sample['emo_embed'], global_steps=300000, infer=True)
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out)
        wav_out = wav_out.squeeze().cpu().numpy()
        return wav_out




if __name__ == '__main__':
    inp = {
        'text': 'here we go',
        'ref_audio': 'assets/0011_001570.wav'
    }
    GenerSpeechInfer.example_run(inp)
