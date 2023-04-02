import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from modules.portaspeech.portaspeech import PortaSpeech
from modules.syntaspeech.multi_window_disc import Discriminator
from tasks.tts.fs2 import FastSpeech2Task
from utils.hparams import hparams
from utils.tts_utils import get_focus_rate, get_phone_coverage_rate, get_diagonal_focus_rate, mel2token_to_dur
from utils import num_params, tensors_to_scalars
from utils.pitch_utils import denorm_f0, norm_f0
from data_gen.tts.data_gen_utils import get_pitch
from utils.dtw import dtw as DTW

from utils.plot import spec_to_figure
from utils.text.text_encoder import build_token_encoder


class PortaSpeechAdvTask(FastSpeech2Task):
    def __init__(self):
        super().__init__()
        data_dir = hparams['binary_data_dir']
        self.word_encoder = build_token_encoder(f'{data_dir}/word_set.json')
        self.build_disc_model()
        self.mse_loss_fn = torch.nn.MSELoss()
        
    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = PortaSpeech(ph_dict_size, word_dict_size, hparams)

        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
        self.dp_params = [p for k, p in self.model.named_parameters() if (('dur_predictor' in k) and p.requires_grad)]
        self.gen_params_except_dp = [p for k, p in self.model.named_parameters() if (('dur_predictor' not in k) and p.requires_grad)]        
        self.bert_params = [p for k, p in self.model.named_parameters() if (('bert' in k) and p.requires_grad)]
        self.gen_params_except_bert_and_dp = [p for k, p in self.model.named_parameters() if ('dur_predictor' not in k) and ('bert' not in k) and p.requires_grad ]

        self.use_bert = True if len(self.bert_params) > 0 else False

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        if hasattr(self.model, 'fvae'):
            for n, m in self.model.fvae.named_children():
                num_params(m, model_name=f'fvae.{n}')

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

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
                                global_step=self.global_step,
                                graph_lst=sample['graph_lst'], 
                                etypes_lst=sample['etypes_lst'],
                                bert_feats=sample.get("bert_feats"),
                                cl_feats=sample.get("cl_feats")
                                )
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
                super(PortaSpeechAdvTask, self).add_dur_loss(output['dur'], sample['mel2ph'], sample['txt_tokens'], losses)
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
                graph_lst=sample['graph_lst'], 
                etypes_lst=sample['etypes_lst'],
                bert_feats=sample.get("bert_feats"),
                cl_feats=sample.get("cl_feats")
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

        with torch.no_grad():
            # calculate word-level abs_dur_error in micro-second
            abs_word_dur_error = F.l1_loss(dur_pred , dur_gt, reduction='none')
            abs_word_dur_error = (abs_word_dur_error * nonpadding).sum() / nonpadding.sum()
            abs_word_dur_error = abs_word_dur_error * hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
            losses['abs_word_dur_error'] = abs_word_dur_error
            # calculate word-level abs_dur_error in second
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            abs_sent_dur_error = F.l1_loss(sent_dur_p, sent_dur_g, reduction='mean').mean()
            abs_sent_dur_error = abs_sent_dur_error * hparams['hop_size'] / hparams['audio_sample_rate']
            losses['abs_sent_dur_error'] = abs_sent_dur_error

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            valid_results = self.save_valid_result(sample, batch_idx, model_out)
            wav_gt = valid_results['wav_gt']
            mel_gt = valid_results['mel_gt']
            wav_pred = valid_results['wav_pred']
            mel_pred = valid_results['mel_pred']
            f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
            f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
            manhattan_distance = lambda x, y: np.abs(x - y)
            dist, cost, acc, path = DTW(f0_pred_, f0_gt_, manhattan_distance)
            outputs['losses']['f0_dtw'] = dist / len(f0_gt_)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        f0_gt = None
        mel_out = model_out['mel_out']
        if sample.get('f0') is not None:
            f0_gt = denorm_f0(sample['f0'][0].cpu(), sample['uv'][0].cpu())
        self.plot_mel(batch_idx, sample['mels'], mel_out, f0s=f0_gt)
        
        # if self.global_step > 0:
        wav_pred = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0_gt)
        self.logger.add_audio(f'wav_val_{batch_idx}', wav_pred, self.global_step, sr)
        # with gt duration
        model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
        dur_info = self.get_plot_dur_info(sample, model_out)
        del dur_info['dur_pred']
        wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
        self.logger.add_audio(f'wav_gdur_{batch_idx}', wav_pred, self.global_step, sr)
        self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_gdur_{batch_idx}',
                        dur_info=dur_info, f0s=f0_gt)

        # with pred duration
        if not hparams['use_gt_dur']:
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
            dur_info = self.get_plot_dur_info(sample, model_out)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_pdur_{batch_idx}',
                            dur_info=dur_info, f0s=f0_gt)
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_pdur_{batch_idx}', wav_pred, self.global_step, sr)
        # gt wav
        mel_gt = sample['mels'][0].cpu()
        wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
        if self.global_step <= hparams['valid_infer_interval']:
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)
        
        # add attn plot
        if self.global_step > 0 and hparams['dur_level'] == 'word':
            self.logger.add_figure(f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)

        return {'wav_gt': wav_gt, 'wav_pred': wav_pred, 'mel_gt': mel_gt, 'mel_pred': model_out['mel_out'][0].cpu()}

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
        
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.dp_params, hparams['clip_grad_norm'])
            if self.use_bert:
                nn.utils.clip_grad_norm_(self.bert_params, hparams['clip_grad_norm'])
                nn.utils.clip_grad_norm_(self.gen_params_except_bert_and_dp, hparams['clip_grad_norm'])
            else:
                nn.utils.clip_grad_norm_(self.gen_params_except_dp, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])

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
        # save f0 for pitch dtw
        f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
        f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
        np.save(f'{gen_dir}/f0/{item_name}.npy', f0_pred_)
        np.save(f'{gen_dir}/f0/{item_name}_gt.npy', f0_gt_)

        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }
