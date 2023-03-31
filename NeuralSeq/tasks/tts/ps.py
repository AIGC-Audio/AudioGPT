import os
import torch
import torch.nn.functional as F
from torch import nn

from modules.portaspeech.portaspeech import PortaSpeech
from tasks.tts.fs2 import FastSpeech2Task
from utils.tts_utils import mel2token_to_dur
from utils.hparams import hparams
from utils.tts_utils import get_focus_rate, get_phone_coverage_rate, get_diagonal_focus_rate
from utils import num_params
import numpy as np

from utils.plot import spec_to_figure
from data_gen.tts.data_gen_utils import build_token_encoder


class PortaSpeechTask(FastSpeech2Task):
    def __init__(self):
        super().__init__()
        data_dir = hparams['binary_data_dir']
        self.word_encoder = build_token_encoder(f'{data_dir}/word_set.json')

    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = PortaSpeech(ph_dict_size, word_dict_size, hparams)

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        if hasattr(self.model, 'fvae'):
            for n, m in self.model.fvae.named_children():
                num_params(m, model_name=f'fvae.{n}')

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']
        word_tokens = sample['word_tokens']
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        if not infer:
            output = self.model(txt_tokens, word_tokens,
                                ph2word=sample['ph2word'],
                                mel2word=sample['mel2word'],
                                mel2ph=sample['mel2ph'],
                                word_len=sample['word_lengths'].max(),
                                tgt_mels=sample['mels'],
                                pitch=sample.get('pitch'),
                                spk_embed=spk_embed,
                                spk_id=spk_id,
                                infer=False,
                                global_step=self.global_step)
            losses = {}
            losses['kl_v'] = output['kl'].detach()
            losses_kl = output['kl']
            losses_kl = torch.clamp(losses_kl, min=hparams['kl_min'])
            losses_kl = min(self.global_step / hparams['kl_start_steps'], 1) * losses_kl
            losses_kl = losses_kl * hparams['lambda_kl']
            losses['kl'] = losses_kl
            self.add_mel_loss(output['mel_out'], sample['mels'], losses)
            if hparams['dur_level'] == 'word':
                self.add_dur_loss(
                    output['dur'], sample['mel2word'], sample['word_lengths'], sample['txt_tokens'], losses)
                self.get_attn_stats(output['attn'], sample, losses)
            else:
                super(PortaSpeechTask, self).add_dur_loss(output['dur'], sample['mel2ph'], sample['txt_tokens'], losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            output = self.model(
                txt_tokens, word_tokens,
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                pitch=sample.get('pitch'),
                mel2ph=sample['mel2ph'] if use_gt_dur else None,
                mel2word=sample['mel2word'] if use_gt_dur else None,
                tgt_mels=sample['mels'],
                infer=True,
                spk_embed=spk_embed,
                spk_id=spk_id,
            )
            return output

    def add_dur_loss(self, dur_pred, mel2token, word_len, txt_tokens, losses=None):
        T = word_len.max()
        dur_gt = mel2token_to_dur(mel2token, T).float()
        nonpadding = (torch.arange(T).to(dur_pred.device)[None, :] < word_len[:, None]).float()
        dur_pred = dur_pred * nonpadding
        dur_gt = dur_gt * nonpadding
        wdur = F.l1_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        wdur = (wdur * nonpadding).sum() / nonpadding.sum()
        if hparams['lambda_word_dur'] > 0:
            losses['wdur'] = wdur * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.l1_loss(sent_dur_p, sent_dur_g, reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def validation_step(self, sample, batch_idx):
        return super().validation_step(sample, batch_idx)

    def save_valid_result(self, sample, batch_idx, model_out):
        super(PortaSpeechTask, self).save_valid_result(sample, batch_idx, model_out)
        if self.global_step > 0 and hparams['dur_level'] == 'word':
            self.logger.add_figure(f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)

    def get_attn_stats(self, attn, sample, logging_outputs, prefix=''):
        # diagonal_focus_rate
        txt_lengths = sample['txt_lengths'].float()
        mel_lengths = sample['mel_lengths'].float()
        src_padding_mask = sample['txt_tokens'].eq(0)
        target_padding_mask = sample['mels'].abs().sum(-1).eq(0)
        src_seg_mask = sample['txt_tokens'].eq(self.seg_idx)
        attn_ks = txt_lengths.float() / mel_lengths.float()

        focus_rate = get_focus_rate(attn, src_padding_mask, target_padding_mask).mean().data
        phone_coverage_rate = get_phone_coverage_rate(
            attn, src_padding_mask, src_seg_mask, target_padding_mask).mean()
        diagonal_focus_rate, diag_mask = get_diagonal_focus_rate(
            attn, attn_ks, mel_lengths, src_padding_mask, target_padding_mask)
        logging_outputs[f'{prefix}fr'] = focus_rate.mean().data
        logging_outputs[f'{prefix}pcr'] = phone_coverage_rate.mean().data
        logging_outputs[f'{prefix}dfr'] = diagonal_focus_rate.mean().data

    def get_plot_dur_info(self, sample, model_out):
        if hparams['dur_level'] == 'word':
            T_txt = sample['word_lengths'].max()
            dur_gt = mel2token_to_dur(sample['mel2word'], T_txt)[0]
            dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
            txt = sample['ph_words'][0].split(" ")
        else:
            T_txt = sample['txt_tokens'].shape[1]
            dur_gt = mel2token_to_dur(sample['mel2ph'], T_txt)[0]
            dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
            txt = self.token_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
            txt = txt.split(" ")
        return {'dur_gt': dur_gt, 'dur_pred': dur_pred, 'txt': txt}

    def build_optimizer(self, model):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return self.optimizer

    def build_scheduler(self, optimizer):
        return FastSpeechTask.build_scheduler(self, optimizer)

    ############
    # infer
    ############
    def test_start(self):
        super().test_start()
        if hparams.get('save_attn', False):
            os.makedirs(f'{self.gen_dir}/attn', exist_ok=True)
        self.model.store_inverse_all()

    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        mel2ph = sample['mel2ph'][0].cpu().numpy()
        mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
        if hparams.get('save_attn', False):
            attn = outputs['attn'][0].cpu().numpy()
            np.save(f'{gen_dir}/attn/{item_name}.npy', attn)
        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }
