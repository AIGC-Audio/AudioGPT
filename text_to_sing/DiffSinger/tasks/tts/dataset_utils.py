from utils.cwt import get_lf0_cwt
import torch.optim
import torch.utils.data
import importlib
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
import numpy as np
from tasks.base_task import BaseDataset
import torch
import torch.optim
import torch.utils.data
import utils
import torch.distributions
from utils.hparams import hparams
from utils.pitch_utils import norm_interp_f0
from resemblyzer import VoiceEncoder
import json
from data_gen.tts.data_gen_utils import build_phone_encoder

class BaseTTSDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.ext_mel2ph = None

        def load_size():
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')

        if prefix == 'test':
            if test_items is not None:
                self.indexed_ds, self.sizes = test_items, test_sizes
            else:
                load_size()
            if hparams['num_test_samples'] > 0:
                self.avail_idxs = [x for x in range(hparams['num_test_samples']) \
                                   if x < len(self.sizes)]
                if len(hparams['test_ids']) > 0:
                    self.avail_idxs = hparams['test_ids'] + self.avail_idxs
            else:
                self.avail_idxs = list(range(len(self.sizes)))
        else:
            load_size()
            self.avail_idxs = list(range(len(self.sizes)))

        if hparams['min_frames'] > 0:
            self.avail_idxs = [
                x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
        self.sizes = [self.sizes[i] for i in self.avail_idxs]

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
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
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

        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch


class FastSpeechDataset(BaseTTSDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(prefix, shuffle, test_items, test_sizes, data_dir)
        self.f0_mean, self.f0_std = hparams.get('f0_mean', None), hparams.get('f0_std', None)
        if prefix == 'test' and hparams['test_input_dir'] != '':
            self.data_dir = hparams['test_input_dir']
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
            self.indexed_ds = sorted(self.indexed_ds, key=lambda item: item['item_name'])
            items = {}
            for i in range(len(self.indexed_ds)):
                speaker = self.indexed_ds[i]['item_name'].split('_')[0]
                if speaker not in items.keys():
                    items[speaker] = [i]
                else:
                    items[speaker].append(i)
            sort_item = sorted(items.values(), key=lambda item_pre_speaker: len(item_pre_speaker), reverse=True)
            self.avail_idxs = [n for a in sort_item for n in a][:hparams['num_test_samples']]
            self.indexed_ds, self.sizes = self.load_test_inputs()
            self.avail_idxs = [i for i in range(hparams['num_test_samples'])]

        if hparams['pitch_type'] == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))

    def __getitem__(self, index):
        sample = super(FastSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        max_frames = hparams['max_frames']
        spec = sample['mel']
        T = spec.shape[0]
        phone = sample['txt_token']
        sample['energy'] = (spec.exp() ** 2).sum(-1).sqrt()
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
            if hparams['pitch_type'] == 'cwt':
                cwt_spec = torch.Tensor(item['cwt_spec'])[:max_frames]
                f0_mean = item.get('f0_mean', item.get('cwt_mean'))
                f0_std = item.get('f0_std', item.get('cwt_std'))
                sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
            elif hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None, hparams)
                f0_phlevel_sum = torch.zeros_like(phone).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(phone).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph, hparams)
        else:
            f0 = uv = torch.zeros_like(mel2ph)
            pitch = None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        batch = super(FastSpeechDataset, self).collater(samples)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples]) if samples[0]['pitch'] is not None else None
        uv = utils.collate_1d([s['uv'] for s in samples])
        energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        batch.update({
            'mel2ph': mel2ph,
            'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = utils.collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        return batch

    def load_test_inputs(self):
        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = json.load(open(ph_set_fn, 'r'))
        print("| phone set: ", ph_set)
        phone_encoder = build_phone_encoder(hparams['binary_data_dir'])
        word_encoder = None
        voice_encoder = VoiceEncoder().cuda()
        encoder = [phone_encoder, word_encoder]
        sizes = []
        items = []
        for i in range(len(self.avail_idxs)):
            item = self._get_item(i)

            item2tgfn = f"{hparams['test_input_dir'].replace('binary', 'processed')}/mfa_outputs/{item['item_name']}.TextGrid"
            item = binarizer_cls.process_item(item['item_name'], item['ph'], item['txt'], item2tgfn,
                                              item['wav_fn'], item['spk_id'], encoder, hparams['binarization_args'])
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                if hparams['binarization_args']['with_spk_embed'] else None  # 判断是否保存embedding文件
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

class FastSpeechWordDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechWordDataset, self).__getitem__(index)
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        sample["ph_words"] = item["ph_words"]
        sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
        sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
        sample["ph2word"] = torch.LongTensor(item['ph2word'][:hparams['max_input_tokens']])
        return sample

    def collater(self, samples):
        batch = super(FastSpeechWordDataset, self).collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = utils.collate_1d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = utils.collate_1d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = utils.collate_1d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        return batch
