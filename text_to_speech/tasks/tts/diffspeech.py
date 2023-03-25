import torch

from text_to_speech.modules.tts.diffspeech.shallow_diffusion_tts import GaussianDiffusion
from tasks.tts.fs2_orig import FastSpeech2OrigTask

import utils
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.commons.ckpt_utils import load_ckpt
from text_to_speech.utils.audio.pitch.utils import denorm_f0


class DiffSpeechTask(FastSpeech2OrigTask):
    def build_tts_model(self):
        # get min and max
        # import torch
        # from tqdm import tqdm
        # v_min = torch.ones([80]) * 100
        # v_max = torch.ones([80]) * -100
        # for i, ds in enumerate(tqdm(self.dataset_cls('train'))):
        #     v_max = torch.max(torch.max(ds['mel'].reshape(-1, 80), 0)[0], v_max)
        #     v_min = torch.min(torch.min(ds['mel'].reshape(-1, 80), 0)[0], v_min)
        #     if i % 100 == 0:
        #         print(i, v_min, v_max)
        # print('final', v_min, v_max)
        dict_size = len(self.token_encoder)
        self.model = GaussianDiffusion(dict_size, hparams)
        if hparams['fs2_ckpt'] != '':
            load_ckpt(self.model.fs2, hparams['fs2_ckpt'], 'model', strict=True)
        # for k, v in self.model.fs2.named_parameters():
        #     if 'predictor' not in k:
        #         v.requires_grad = False
        # or
        for k, v in self.model.fs2.named_parameters():
            v.requires_grad = False

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        if not infer:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph = sample['mel2ph']  # [B, T_s]
            f0 = sample.get('f0')
            uv = sample.get('uv')
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                ref_mels=target, f0=f0, uv=uv, infer=False)
            losses = {}
            if 'diff_loss' in output:
                losses['mel'] = output['diff_loss']
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            mel2ph, uv, f0 = None, None, None
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                ref_mels=None, f0=f0, uv=uv, infer=True)
            return output

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        f0_gt = None
        # mel_out = model_out['mel_out']
        if sample.get('f0') is not None:
            f0_gt = denorm_f0(sample['f0'][0].cpu(), sample['uv'][0].cpu())
        # self.plot_mel(batch_idx, sample['mels'], mel_out, f0s=f0_gt)
        if self.global_step > 0:
            # wav_pred = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0_gt)
            # self.logger.add_audio(f'wav_val_{batch_idx}', wav_pred, self.global_step, sr)
            # with gt duration
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
            dur_info = self.get_plot_dur_info(sample, model_out)
            del dur_info['dur_pred']
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_gdur_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'diffmel_gdur_{batch_idx}',
                          dur_info=dur_info, f0s=f0_gt)
            self.plot_mel(batch_idx, sample['mels'], model_out['fs2_mel'][0], f'fs2mel_gdur_{batch_idx}',
                          dur_info=dur_info, f0s=f0_gt)  # gt mel vs. fs2 mel

            # with pred duration
            if not hparams['use_gt_dur']:
                model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
                dur_info = self.get_plot_dur_info(sample, model_out)
                self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_pdur_{batch_idx}',
                              dur_info=dur_info, f0s=f0_gt)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
                self.logger.add_audio(f'wav_pdur_{batch_idx}', wav_pred, self.global_step, sr)
        # gt wav
        if self.global_step <= hparams['valid_infer_interval']:
            mel_gt = sample['mels'][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)
