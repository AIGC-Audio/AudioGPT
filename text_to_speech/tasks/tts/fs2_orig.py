import torch
import torch.nn.functional as F
from text_to_speech.modules.tts.fs2_orig import FastSpeech2Orig
from tasks.tts.dataset_utils import FastSpeechDataset
from tasks.tts.fs import FastSpeechTask
from text_to_speech.utils.commons.dataset_utils import collate_1d, collate_2d
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.plot.plot import spec_to_figure
import numpy as np


class FastSpeech2OrigDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = hparams.get('pitch_type')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        sample['energy'] = (mel.exp() ** 2).sum(-1).sqrt()
        if hparams['use_pitch_embed'] and self.pitch_type == 'cwt':
            cwt_spec = torch.Tensor(item['cwt_spec'])[:T]
            f0_mean = item.get('f0_mean', item.get('cwt_mean'))
            f0_std = item.get('f0_std', item.get('cwt_std'))
            sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super().collater(samples)
        if hparams['use_pitch_embed']:
            energy = collate_1d([s['energy'] for s in samples], 0.0)
        else:
            energy = None
        batch.update({'energy': energy})
        if self.pitch_type == 'cwt':
            cwt_spec = collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        return batch


class FastSpeech2OrigTask(FastSpeechTask):
    def __init__(self):
        super(FastSpeech2OrigTask, self).__init__()
        self.dataset_cls = FastSpeech2OrigDataset

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = FastSpeech2Orig(dict_size, hparams)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        if not infer:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph = sample['mel2ph']  # [B, T_s]
            f0 = sample.get('f0')
            uv = sample.get('uv')
            energy = sample.get('energy')
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, energy=energy, infer=False)
            losses = {}
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            if hparams['use_energy_embed']:
                self.add_energy_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph, uv, f0, energy = None, None, None, None
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            use_gt_energy = kwargs.get('infer_use_gt_energy', hparams['use_gt_energy'])
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            if use_gt_energy:
                energy = sample['energy']
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, energy=energy, infer=True)
            return output

    def save_valid_result(self, sample, batch_idx, model_out):
        super(FastSpeech2OrigTask, self).save_valid_result(sample, batch_idx, model_out)
        self.plot_cwt(batch_idx, model_out['cwt'], sample['cwt_spec'])

    def plot_cwt(self, batch_idx, cwt_out, cwt_gt=None):
        if len(cwt_out.shape) == 3:
            cwt_out = cwt_out[0]
        if isinstance(cwt_out, torch.Tensor):
            cwt_out = cwt_out.cpu().numpy()
        if cwt_gt is not None:
            if len(cwt_gt.shape) == 3:
                cwt_gt = cwt_gt[0]
            if isinstance(cwt_gt, torch.Tensor):
                cwt_gt = cwt_gt.cpu().numpy()
            cwt_out = np.concatenate([cwt_out, cwt_gt], -1)
        name = f'cwt_val_{batch_idx}'
        self.logger.add_figure(name, spec_to_figure(cwt_out), self.global_step)

    def add_pitch_loss(self, output, sample, losses):
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            uv = sample['uv']
            mel2ph = sample['mel2ph']
            f0_std = sample['f0_std']
            cwt_pred = output['cwt'][:, :, :10]
            f0_mean_pred = output['f0_mean']
            f0_std_pred = output['f0_std']
            nonpadding = (mel2ph != 0).float()
            losses['C'] = F.l1_loss(cwt_pred, cwt_spec) * hparams['lambda_f0']
            if hparams['use_uv']:
                assert output['cwt'].shape[-1] == 11
                uv_pred = output['cwt'][:, :, -1]
                losses['uv'] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction='none')
                                * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
            losses['f0_mean'] = F.l1_loss(f0_mean_pred, f0_mean) * hparams['lambda_f0']
            losses['f0_std'] = F.l1_loss(f0_std_pred, f0_std) * hparams['lambda_f0']
        else:
            super(FastSpeech2OrigTask, self).add_pitch_loss(output, sample, losses)

    def add_energy_loss(self, output, sample, losses):
        energy_pred, energy = output['energy_pred'], sample['energy']
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        losses['e'] = loss
