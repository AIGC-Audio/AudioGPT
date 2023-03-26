import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.ckpt_utils import load_ckpt, get_last_checkpoint
from utils.hparams import hparams
from modules.GenerSpeech.model.generspeech import GenerSpeech
import os
import numpy as np
from functools import partial
from utils.hparams import set_hparams
from utils.hparams import hparams as hp

class GenerSpeechInfer(BaseTTSInfer):
    def build_model(self):
        model = GenerSpeech(self.ph_encoder)
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')
        return model

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
