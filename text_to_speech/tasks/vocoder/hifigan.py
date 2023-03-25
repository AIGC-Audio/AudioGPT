import torch.nn.functional as F
from torch import nn

from text_to_speech.modules.vocoder.hifigan.hifigan import HifiGanGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    generator_loss, feature_loss, discriminator_loss
from text_to_speech.modules.vocoder.hifigan.mel_utils import mel_spectrogram
from text_to_speech.modules.vocoder.hifigan.stft_loss import MultiResolutionSTFTLoss
from tasks.vocoder.vocoder_base import VocoderBaseTask
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.nn.model_utils import print_arch


class HifiGanTask(VocoderBaseTask):
    def build_model(self):
        self.model_gen = HifiGanGenerator(hparams)
        self.model_disc = nn.ModuleDict()
        self.model_disc['mpd'] = MultiPeriodDiscriminator()
        self.model_disc['msd'] = MultiScaleDiscriminator()
        self.stft_loss = MultiResolutionSTFTLoss()
        print_arch(self.model_gen)
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], 'model_gen', 'model_gen', force=True, strict=True)
            self.load_ckpt(hparams['load_ckpt'], 'model_disc', 'model_disc', force=True, strict=True)
        return self.model_gen

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mel = sample['mels']
        y = sample['wavs']
        f0 = sample['f0']
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            y_ = self.model_gen(mel, f0)
            y_mel = mel_spectrogram(y.squeeze(1), hparams).transpose(1, 2)
            y_hat_mel = mel_spectrogram(y_.squeeze(1), hparams).transpose(1, 2)
            loss_output['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']
            _, y_p_hat_g, fmap_f_r, fmap_f_g = self.model_disc['mpd'](y, y_, mel)
            _, y_s_hat_g, fmap_s_r, fmap_s_g = self.model_disc['msd'](y, y_, mel)
            loss_output['a_p'] = generator_loss(y_p_hat_g) * hparams['lambda_adv']
            loss_output['a_s'] = generator_loss(y_s_hat_g) * hparams['lambda_adv']
            if hparams['use_fm_loss']:
                loss_output['fm_f'] = feature_loss(fmap_f_r, fmap_f_g)
                loss_output['fm_s'] = feature_loss(fmap_s_r, fmap_s_g)
            if hparams['use_ms_stft']:
                loss_output['sc'], loss_output['mag'] = self.stft_loss(y.squeeze(1), y_.squeeze(1))
            self.y_ = y_.detach()
            self.y_mel = y_mel.detach()
            self.y_hat_mel = y_hat_mel.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            y_ = self.y_
            # MPD
            y_p_hat_r, y_p_hat_g, _, _ = self.model_disc['mpd'](y, y_.detach(), mel)
            loss_output['r_p'], loss_output['f_p'] = discriminator_loss(y_p_hat_r, y_p_hat_g)
            # MSD
            y_s_hat_r, y_s_hat_g, _, _ = self.model_disc['msd'](y, y_.detach(), mel)
            loss_output['r_s'], loss_output['f_s'] = discriminator_loss(y_s_hat_r, y_s_hat_g)
        total_loss = sum(loss_output.values())
        return total_loss, loss_output
