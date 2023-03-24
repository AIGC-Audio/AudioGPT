import matplotlib
matplotlib.use('Agg')
from tasks.base_task import data_loader
from tasks.tts.fs2 import FastSpeech2Task
from tasks.tts.dataset_utils import FastSpeechDataset, BaseTTSDataset
import glob
import importlib
from utils.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
from inference.base_tts_infer import load_data_preprocessor
from data_gen.tts.emotion import inference as EmotionEncoder
from data_gen.tts.emotion.inference import embed_utterance as Embed_utterance
from data_gen.tts.emotion.inference import preprocess_wav
from tqdm import tqdm
from utils.hparams import hparams
from data_gen.tts.data_gen_utils import build_phone_encoder, build_word_encoder
import random
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from utils.indexed_datasets import IndexedDataset
from resemblyzer import VoiceEncoder
import torch.distributions
import numpy as np
import utils
import os



class GenerSpeech_dataset(BaseTTSDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(prefix, shuffle, test_items, test_sizes, data_dir)
        self.f0_mean, self.f0_std = hparams.get('f0_mean', None), hparams.get('f0_std', None)
        if prefix == 'valid':
            indexed_ds = IndexedDataset(f'{self.data_dir}/train')
            sizes = np.load(f'{self.data_dir}/train_lengths.npy')
            index = [i for i in range(len(indexed_ds))]
            random.shuffle(index)
            index = index[:300]
            self.sizes = sizes[index]
            self.indexed_ds = []
            for i in index:
                self.indexed_ds.append(indexed_ds[i])
            self.avail_idxs = list(range(len(self.sizes)))
            if hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if prefix == 'test' and hparams['test_input_dir'] != '':
            self.preprocessor, self.preprocess_args = load_data_preprocessor()
            self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            self.avail_idxs = [i for i, _ in enumerate(self.sizes)]


    def load_test_inputs(self, test_input_dir):
        inp_wav_paths = sorted(glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3'))
        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)

        phone_encoder = build_phone_encoder(hparams['binary_data_dir'])
        word_encoder = build_word_encoder(hparams['binary_data_dir'])
        voice_encoder = VoiceEncoder().cuda()

        encoder = [phone_encoder, word_encoder]
        sizes = []
        items = []
        EmotionEncoder.load_model(hparams['emotion_encoder_path'])
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args

        for wav_fn in tqdm(inp_wav_paths):
            item_name = wav_fn[len(test_input_dir) + 1:].replace("/", "_")
            spk_id = emotion = 0
            item2tgfn = wav_fn.replace('.wav', '.TextGrid') # prepare textgrid alignment
            txtpath = wav_fn.replace('.wav', '.txt')  # prepare text
            with open(txtpath, 'r') as f:
                text_raw = f.readlines()
                f.close()
            ph, txt = preprocessor.txt_to_ph(preprocessor.txt_processor, text_raw[0], preprocess_args)

            item = binarizer_cls.process_item(item_name, ph, txt, item2tgfn, wav_fn, spk_id, emotion, encoder, hparams['binarization_args'])
            item['emo_embed'] = Embed_utterance(preprocess_wav(item['wav_fn']))
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        spec = sample['mel']
        T = spec.shape[0]
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T] if 'mel2ph' in item else None
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            if hparams.get('normalize_pitch', False):
                f0 = item["f0"]
                if len(f0 > 0) > 0 and f0[f0 > 0].std() > 0:
                    f0[f0 > 0] = (f0[f0 > 0] - f0[f0 > 0].mean()) / f0[f0 > 0].std() * hparams['f0_std'] + \
                                 hparams['f0_mean']
                    f0[f0 > 0] = f0[f0 > 0].clip(min=60, max=500)
                pitch = f0_to_coarse(f0)
                pitch = torch.LongTensor(pitch[:max_frames])
            else:
                pitch = torch.LongTensor(item.get("pitch"))[:max_frames] if "pitch" in item else None
            f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
        else:
            f0 = uv = torch.zeros_like(mel2ph)
            pitch = None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        sample["emotion"] = item['emotion']
        sample["emo_embed"] = torch.Tensor(item['emo_embed'])

        if hparams.get('use_word', False):
            sample["ph_words"] = item["ph_words"]
            sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
            sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
            sample["ph2word"] = torch.LongTensor(item['ph2word'][:hparams['max_input_tokens']])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
        }

        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples]) if samples[0]['pitch'] is not None else None
        uv = utils.collate_1d([s['uv'] for s in samples])
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) if samples[0]['mel2ph'] is not None else None
        batch.update({
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        spk_embed = torch.stack([s['spk_embed'] for s in samples])
        batch['spk_embed'] = spk_embed
        emo_embed = torch.stack([s['emo_embed'] for s in samples])
        batch['emo_embed'] = emo_embed

        if hparams.get('use_word', False):
            ph_words = [s['ph_words'] for s in samples]
            batch['ph_words'] = ph_words
            word_tokens = utils.collate_1d([s['word_tokens'] for s in samples], 0)
            batch['word_tokens'] = word_tokens
            mel2word = utils.collate_1d([s['mel2word'] for s in samples], 0)
            batch['mel2word'] = mel2word
            ph2word = utils.collate_1d([s['ph2word'] for s in samples], 0)
            batch['ph2word'] = ph2word
        return batch