import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from text_to_speech.utils.audio.pitch.utils import norm_interp_f0, denorm_f0
from text_to_speech.utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from text_to_speech.utils.commons.indexed_datasets import IndexedDataset
from text_to_speech.utils.commons.hparams import hparams
import random 


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from text_to_speech.utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            try:
                self.sizes = [self.sizes[i] for i in self.avail_idxs]
            except:
                tmp_sizes = []
                for i in self.avail_idxs:
                    try:
                        tmp_sizes.append(self.sizes[i])
                    except:
                        continue
                self.sizes = tmp_sizes
                
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
        ph_token = torch.LongTensor(item['ph_token'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = int(item['spk_id'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        ids = [s['id'] for s in samples]
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = collate_1d_or_2d([s['txt_token'] for s in samples], 0)
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': ids,
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


class FastSpeechDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        ph_token = sample['txt_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(FastSpeechDataset, self).collater(samples)
        hparams = self.hparams
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
            uv = collate_1d_or_2d([s['uv'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        batch.update({
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        return batch

class FastSpeechWordDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        # BERT contrastive loss & mlm loss
        # from transformers import AutoTokenizer
        # if hparams['ds_name'] in ['ljspeech', 'libritts']:
        #     self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # elif hparams['ds_name'] == 'biaobei':
        #     self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        # else:
        #     raise NotImplementedError()
        # self.mlm_probability = 0.15
        # if hparams.get("cl_ds_name") is None:
        #     pass
        # elif hparams['cl_ds_name'] == "wiki":
        #     from experimental_yerfor.simcse_datasets import WikiDataset
        #     self.cl_dataset = WikiDataset(prefix=prefix)
        #     shuffle = True if prefix == 'train' else False
        #     endless = True
        #     num_workers = None if prefix == 'train' else 0
        #     self.cl_dataloader = self.cl_dataset.build_dataloader(shuffle=shuffle, max_tokens=hparams.get("cl_max_tokens", 3200),
        #         max_sentences=hparams.get("cl_max_sentences", 64), endless=endless, num_workers=num_workers)
        #     self.cl_dl_iter = iter(self.cl_dataloader)
        # elif hparams['cl_ds_name'] == "nli":
        #     from experimental_yerfor.simcse_datasets import NLIDataset
        #     self.cl_dataset = NLIDataset(prefix=prefix)
        #     shuffle = True if prefix == 'train' else False
        #     endless = True
        #     num_workers = None if prefix == 'train' else 0
        #     self.cl_dataloader = self.cl_dataset.build_dataloader(shuffle=shuffle, max_tokens=hparams.get("cl_max_tokens", 4800),
        #         max_sentences=hparams.get("cl_max_sentences", 128), endless=endless, num_workers=num_workers)
        #     self.cl_dl_iter = iter(self.cl_dataloader)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        if 'word' in item:
            sample['words'] = item['word']
            sample["ph_words"] = item["ph_gb_word"]
            sample["word_tokens"] = torch.LongTensor(item["word_token"])
        else:
            sample['words'] = item['words']
            sample["ph_words"] = " ".join(item["ph_words"])
            sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
        sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
        sample["ph2word"] = torch.LongTensor(item['ph2word'][:self.hparams['max_input_tokens']])

        # SyntaSpeech related features
        # sample['dgl_graph'] = item['dgl_graph']
        # sample['edge_types'] = item['edge_types']

        # BERT related features
        # sample['bert_token'] = item['bert_token']
        # sample['bert_input_ids'] = torch.LongTensor(item['bert_input_ids'])
        # sample['bert_token2word'] = torch.LongTensor(item['bert_token2word'])
        # sample['bert_attention_mask'] = torch.LongTensor(item['bert_attention_mask'])
        # sample['bert_token_type_ids'] = torch.LongTensor(item['bert_token_type_ids'])

        return sample

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        batch = super().collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = collate_1d_or_2d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = collate_1d_or_2d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['words'] = [s['words'] for s in samples]
        batch['word_lengths'] = torch.LongTensor([len(s['word_tokens']) for s in samples])
        if self.hparams['use_word_input']: # always False
            batch['txt_tokens'] = batch['word_tokens']
            batch['txt_lengths'] = torch.LongTensor([s['word_tokens'].numel() for s in samples])
            batch['mel2ph'] = batch['mel2word']
        
        # SyntaSpeech
        # graph_lst, etypes_lst = [], [] # new features for Graph-based SDP
        # for s in samples:
        #     graph_lst.append(s['dgl_graph'])
        #     etypes_lst.append(s['edge_types'])
        # batch.update({
        #     'graph_lst': graph_lst,
        #     'etypes_lst': etypes_lst,
        # })

        # BERT
        # batch['bert_feats'] = {}
        # batch['bert_feats']['bert_tokens'] = [s['bert_token'] for s in samples]
        # bert_input_ids = collate_1d_or_2d([s['bert_input_ids'] for s in samples], 0)
        # batch['bert_feats']['bert_input_ids'] = bert_input_ids
        # bert_token2word = collate_1d_or_2d([s['bert_token2word'] for s in samples], 0)
        # batch['bert_feats']['bert_token2word'] = bert_token2word
        # bert_attention_mask = collate_1d_or_2d([s['bert_attention_mask'] for s in samples], 0)
        # batch['bert_feats']['bert_attention_mask'] = bert_attention_mask
        # bert_token_type_ids = collate_1d_or_2d([s['bert_token_type_ids'] for s in samples], 0)
        # batch['bert_feats']['bert_token_type_ids'] = bert_token_type_ids
        
        # BERT contrastive loss & mlm loss & electra loss
        # if hparams.get("cl_ds_name") is None:
        #     batch['cl_feats'] = {}
        #     batch['cl_feats']['cl_input_ids'] = batch['bert_feats']['bert_input_ids'].unsqueeze(1).repeat([1,2,1])
        #     batch['cl_feats']['cl_token2word'] = batch['bert_feats']['bert_token2word'].unsqueeze(1).repeat([1,2,1])
        #     batch['cl_feats']['cl_attention_mask'] = batch['bert_feats']['bert_attention_mask'].unsqueeze(1).repeat([1,2,1])
        #     batch['cl_feats']['cl_token_type_ids'] = batch['bert_feats']['bert_token_type_ids'].unsqueeze(1).repeat([1,2,1])
        #     bs, _, t = batch['cl_feats']['cl_input_ids'].shape
        #     mlm_input_ids, mlm_labels = self.mask_tokens(batch['bert_feats']['bert_input_ids'].reshape([bs, t]))
        #     batch['cl_feats']["mlm_input_ids"] = mlm_input_ids.reshape([bs, t])
        #     batch['cl_feats']["mlm_labels"] = mlm_labels.reshape([bs, t])
        #     batch['cl_feats']["mlm_attention_mask"] = batch['bert_feats']['bert_attention_mask']
        # elif hparams['cl_ds_name'] in ["wiki", "nli"]:
        #     try:
        #         cl_feats = self.cl_dl_iter.__next__()
        #     except:
        #         self.cl_dl_iter = iter(self.cl_dataloader)
        #         cl_feats = self.cl_dl_iter.__next__()
        #     batch['cl_feats'] = cl_feats
        return batch

    # def mask_tokens(self, inputs, special_tokens_mask=None):
    #     """
    #     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    #     """
    #     inputs = inputs.clone()
    #     labels = inputs.clone()
    #     # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    #     probability_matrix = torch.full(labels.shape, self.mlm_probability)
    #     if special_tokens_mask is None:
    #         special_tokens_mask = [
    #             self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    #         ]
    #         special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    #     else:
    #         special_tokens_mask = special_tokens_mask.bool()

    #     probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    #     masked_indices = torch.bernoulli(probability_matrix).bool()
    #     labels[~masked_indices] = -100  # We only compute loss on masked tokens

    #     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    #     inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    #     # 10% of the time, we replace masked input tokens with random word
    #     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #     random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    #     inputs[indices_random] = random_words[indices_random]

    #     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #     return inputs, labels

