import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from text_to_speech.modules.tts.syntaspeech.multi_window_disc import Discriminator
from tasks.tts.fs import FastSpeechTask
from text_to_speech.modules.tts.fs import FastSpeech

from text_to_speech.utils.audio.align import mel2token_to_dur
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.nn.model_utils import num_params
from text_to_speech.utils.commons.tensor_utils import tensors_to_scalars
from text_to_speech.utils.audio.pitch.utils import denorm_f0, norm_f0
from text_to_speech.utils.audio.pitch_extractors import get_pitch
from text_to_speech.utils.metrics.dtw import dtw as DTW

from text_to_speech.utils.plot.plot import spec_to_figure
from text_to_speech.utils.text.text_encoder import build_token_encoder


class FastSpeechAdvTask(FastSpeechTask):
    def __init__(self):
        super().__init__()
        self.build_disc_model()
        self.mse_loss_fn = torch.nn.MSELoss()
        
    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = FastSpeech(dict_size, hparams)
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
            else:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output


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
        # if self.global_step > 0 and hparams['dur_level'] == 'word':
        #     self.logger.add_figure(f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)

        return {'wav_gt': wav_gt, 'wav_pred': wav_pred, 'mel_gt': mel_gt, 'mel_pred': model_out['mel_out'][0].cpu()}


    def get_plot_dur_info(self, sample, model_out):
        # if hparams['dur_level'] == 'word':
        #     T_txt = sample['word_lengths'].max()
        #     dur_gt = mel2token_to_dur(sample['mel2word'], T_txt)[0]
        #     dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
        #     txt = sample['ph_words'][0].split(" ")
        # else:
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
