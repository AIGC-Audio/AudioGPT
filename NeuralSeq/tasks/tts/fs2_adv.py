from tasks.tts.fs2 import FastSpeech2Task
from modules.syntaspeech.multi_window_disc import Discriminator
from utils.hparams import hparams
from torch import nn
import torch
import torch.optim
import torch.utils.data
import utils


class FastSpeech2AdvTask(FastSpeech2Task):
    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.print_arch(self.model, 'Generator')
        self.build_disc_model()
        if not hasattr(self, 'gen_params'):
            self.gen_params = list(self.model.parameters())
        return self.model

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())
        utils.print_arch(self.mel_disc, model_name='Mel Disc')

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step >= hparams["disc_start_steps"] and \
                     hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            log_outputs, model_out = self.run_model(self.model, sample, return_output=True)
            self.model_out = {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                self.disc_cond = disc_cond = self.model_out['decoder_inp'].detach() \
                    if hparams['use_cond_disc'] else None
                if hparams['mel_loss_no_noise']:
                    self.add_mel_loss(model_out['mel_out_nonoise'], sample['mels'], log_outputs)
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p, disc_cond)
                p_, pc_ = o_['y'], o_['y_c']

                if p_ is not None:
                    log_outputs['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    log_outputs['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                if hparams['rerun_gen']:
                    with torch.no_grad():
                        _, model_out = self.run_model(self.model, sample, return_output=True)
                else:
                    model_out = self.model_out
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)

                o = self.mel_disc(mel_g, self.disc_cond)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p, self.disc_cond)
                p_, pc_ = o_['y'], o_['y_c']

                if p_ is not None:
                    log_outputs["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    log_outputs["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))

                if pc_ is not None:
                    log_outputs["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    log_outputs["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
                   
            if len(log_outputs) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])

        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

    def configure_optimizers(self):
        if not hasattr(self, 'gen_params'):
            self.gen_params = list(self.model.parameters())
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
        self.scheduler = self.build_scheduler({'gen': optimizer_gen, 'disc': optimizer_disc})
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": super().build_scheduler(optimizer['gen']),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["disc"],
                **hparams["discriminator_scheduler_params"]) if optimizer["disc"] is not None else None,
        }

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step)
        else:
            self.scheduler['disc'].step(max(self.global_step - hparams["disc_start_steps"], 1))
