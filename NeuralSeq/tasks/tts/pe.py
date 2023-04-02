import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import os

from tasks.base_task import BaseDataset
from tasks.tts.fs2 import FastSpeech2Task
from modules.fastspeech.pe import PitchExtractor
import utils
from utils.indexed_datasets import IndexedDataset
from utils.hparams import hparams
from utils.plot import f0_to_figure
from utils.pitch_utils import norm_interp_f0, denorm_f0


class PeDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['num_test_samples'] > 0:
                self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
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
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        # mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        # print(item.keys(), item['mel'].shape, spec.shape)
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "mel": spec,
            "pitch": pitch,
            "f0": f0,
            "uv": uv,
            # "mel2ph": mel2ph,
            # "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        # mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
        #     if samples[0]['mel2ph'] is not None else None
        # mel_nonpaddings = utils.collate_1d([s['mel_nonpadding'].float() for s in samples], 0.0)

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'pitch': pitch,
            # 'mel2ph': mel2ph,
            # 'mel_nonpaddings': mel_nonpaddings,
            'f0': f0,
            'uv': uv,
        }
        return batch


class PitchExtractionTask(FastSpeech2Task):
    def __init__(self):
        super().__init__()
        self.dataset_cls = PeDataset

    def build_tts_model(self):
        self.model = PitchExtractor(conv_layers=hparams['pitch_extractor_conv_layers'])

    # def build_scheduler(self, optimizer):
    #     return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)
    def _training_step(self, sample, batch_idx, _):
        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['mels'].size()[0]
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            self.plot_pitch(batch_idx, model_out, sample)
        return outputs

    def run_model(self, model, sample, return_output=False, infer=False):
        f0 = sample['f0']
        uv = sample['uv']
        output = model(sample['mels'])
        losses = {}
        self.add_pitch_loss(output, sample, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def plot_pitch(self, batch_idx, model_out, sample):
        gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
        self.logger.experiment.add_figure(
            f'f0_{batch_idx}',
            f0_to_figure(gt_f0[0], None, model_out['f0_denorm_pred'][0]),
            self.global_step)

    def add_pitch_loss(self, output, sample, losses):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        mel = sample['mels']
        f0 = sample['f0']
        uv = sample['uv']
        # nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
        #     else (sample['txt_tokens'] != 0).float()
        nonpadding = (mel.abs().sum(-1) > 0).float()  # sample['mel_nonpaddings']
        # print(nonpadding[0][-8:], nonpadding.shape)
        self.add_f0_loss(output['pitch_pred'], f0, uv, losses, nonpadding=nonpadding)