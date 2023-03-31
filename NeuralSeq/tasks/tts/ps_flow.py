import torch
from modules.portaspeech.portaspeech_flow import PortaSpeechFlow
from tasks.tts.fs2 import FastSpeech2Task
from tasks.tts.ps import PortaSpeechTask
from utils.pitch_utils import denorm_f0
from utils.hparams import hparams


class PortaSpeechFlowTask(PortaSpeechTask):
    def __init__(self):
        super().__init__()
        self.training_post_glow = False

    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = PortaSpeechFlow(ph_dict_size, word_dict_size, hparams)

    def _training_step(self, sample, batch_idx, opt_idx):
        self.training_post_glow = self.global_step >= hparams['post_glow_training_start'] \
                                  and hparams['use_post_flow']
        if hparams['two_stage'] and \
                ((opt_idx == 0 and self.training_post_glow) or (opt_idx == 1 and not self.training_post_glow)):
            return None
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        if 'postflow' in loss_output and loss_output['postflow'] is None:
            return None
        return total_loss, loss_output

    def run_model(self, sample, infer=False, *args, **kwargs):
        if not infer:
            training_post_glow = self.training_post_glow
            spk_embed = sample.get('spk_embed')
            spk_id = sample.get('spk_ids')
            output = self.model(sample['txt_tokens'],
                                sample['word_tokens'],
                                ph2word=sample['ph2word'],
                                mel2word=sample['mel2word'],
                                mel2ph=sample['mel2ph'],
                                word_len=sample['word_lengths'].max(),
                                tgt_mels=sample['mels'],
                                pitch=sample.get('pitch'),
                                spk_embed=spk_embed,
                                spk_id=spk_id,
                                infer=False,
                                forward_post_glow=training_post_glow,
                                two_stage=hparams['two_stage'],
                                global_step=self.global_step,
                                bert_feats=sample.get('bert_feats'))
            losses = {}
            self.add_mel_loss(output['mel_out'], sample['mels'], losses)
            if (training_post_glow or not hparams['two_stage']) and hparams['use_post_flow']:
                losses['postflow'] = output['postflow']
                losses['l1'] = losses['l1'].detach()
                losses['ssim'] = losses['ssim'].detach()
            if not training_post_glow or not hparams['two_stage'] or not self.training:
                losses['kl'] = output['kl']
                if self.global_step < hparams['kl_start_steps']:
                    losses['kl'] = losses['kl'].detach()
                else:
                    losses['kl'] = torch.clamp(losses['kl'], min=hparams['kl_min'])
                losses['kl'] = losses['kl'] * hparams['lambda_kl']
                if hparams['dur_level'] == 'word':
                    self.add_dur_loss(
                        output['dur'], sample['mel2word'], sample['word_lengths'], sample['txt_tokens'], losses)
                    self.get_attn_stats(output['attn'], sample, losses)
                else:
                    super().add_dur_loss(output['dur'], sample['mel2ph'], sample['txt_tokens'], losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            forward_post_glow = self.global_step >= hparams['post_glow_training_start'] + 1000 \
                                and hparams['use_post_flow']
            spk_embed = sample.get('spk_embed')
            spk_id = sample.get('spk_ids')
            output = self.model(
                sample['txt_tokens'],
                sample['word_tokens'],
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                pitch=sample.get('pitch'),
                mel2ph=sample['mel2ph'] if use_gt_dur else None,
                mel2word=sample['mel2word'] if hparams['profile_infer'] or hparams['use_gt_dur'] else None,
                infer=True,
                forward_post_glow=forward_post_glow,
                spk_embed=spk_embed,
                spk_id=spk_id,
                two_stage=hparams['two_stage'],
                bert_feats=sample.get('bert_feats'))
            return output

    def validation_step(self, sample, batch_idx):
        self.training_post_glow = self.global_step >= hparams['post_glow_training_start'] \
                                  and hparams['use_post_flow']
        return super().validation_step(sample, batch_idx)

    def save_valid_result(self, sample, batch_idx, model_out):
        super(PortaSpeechFlowTask, self).save_valid_result(sample, batch_idx, model_out)
        sr = hparams['audio_sample_rate']
        f0_gt = None
        if sample.get('f0') is not None:
            f0_gt = denorm_f0(sample['f0'][0].cpu(), sample['uv'][0].cpu())
        if self.global_step > 0:
            # save FVAE result
            if hparams['use_post_flow']:
                wav_pred = self.vocoder.spec2wav(model_out['mel_out_fvae'][0].cpu(), f0=f0_gt)
                self.logger.add_audio(f'wav_fvae_{batch_idx}', wav_pred, self.global_step, sr)
                self.plot_mel(batch_idx, sample['mels'], model_out['mel_out_fvae'][0],
                              f'mel_fvae_{batch_idx}', f0s=f0_gt)

    def build_optimizer(self, model):
        if hparams['two_stage'] and hparams['use_post_flow']:
            self.optimizer = torch.optim.AdamW(
                [p for name, p in self.model.named_parameters() if 'post_flow' not in name],
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            self.post_flow_optimizer = torch.optim.AdamW(
                self.model.post_flow.parameters(),
                lr=hparams['post_flow_lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            return [self.optimizer, self.post_flow_optimizer]
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            return [self.optimizer]

    def build_scheduler(self, optimizer):
        return FastSpeech2Task.build_scheduler(self, optimizer[0])