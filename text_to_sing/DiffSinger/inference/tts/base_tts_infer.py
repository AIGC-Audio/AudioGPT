from data_gen.tts.data_gen_utils import is_sil_phoneme
from resemblyzer import VoiceEncoder
from data_gen.tts.data_gen_utils import build_phone_encoder, build_word_encoder
from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.tts_utils import load_data_preprocessor
from vocoders.hifigan import HifiGanGenerator
from data_gen.tts.emotion import inference as EmotionEncoder
from data_gen.tts.emotion.inference import embed_utterance as Embed_utterance
from data_gen.tts.emotion.inference import preprocess_wav
import importlib
import os
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from string import punctuation
import torch
from utils import audio
from utils.ckpt_utils import load_ckpt
from utils.hparams import set_hparams
from utils.hparams import hparams as hp

class BaseTTSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor, self.preprocess_args = load_data_preprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)
        self.asr_processor, self.asr_model = self.build_asr()

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_asr(self):
        # load pretrained model
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  # facebook/wav2vec2-base-960h  wav2vec2-large-960h-lv60-self
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        return processor, model

    def build_vocoder(self):
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=False)
        vocoder = HifiGanGenerator(config)
        load_ckpt(vocoder, base_dir, 'model_gen')
        return vocoder

    def run_vocoder(self, c):
        c = c.transpose(2, 1)
        y = self.vocoder(c)[:, 0]
        return y

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

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls, inp):
        from utils.audio import save_wav

        #set_hparams(print_hparams=False)
        infer_ins = cls(hp)
        out = infer_ins.infer_once(inp)
        os.makedirs('infer_out', exist_ok=True)
        save_wav(out, f'infer_out/{hp["text"]}.wav', hp['audio_sample_rate'])
        print(f'Save at infer_out/{hp["text"]}.wav.')

    def asr(self, file):
        sample_rate = self.hparams['audio_sample_rate']
        audio_input, source_sample_rate = sf.read(file)

        # Resample the wav if needed
        if sample_rate is not None and source_sample_rate != sample_rate:
            audio_input = librosa.resample(audio_input, source_sample_rate, sample_rate)

        # pad input values and return pt tensor
        input_values = self.asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

        # retrieve logits & take argmax
        logits = self.asr_model(input_values.cuda()).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # transcribe
        transcription = self.asr_processor.decode(predicted_ids[0])
        transcription = transcription.rstrip(punctuation)
        return audio_input, transcription