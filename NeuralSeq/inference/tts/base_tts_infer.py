from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.tts_utils import load_data_preprocessor
from vocoders.hifigan import HifiGanGenerator
import os
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from string import punctuation
import torch
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
        raise NotImplementedError

    def input_to_batch(self, item):
        raise NotImplementedError

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