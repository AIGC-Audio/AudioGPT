import torch
from torch import nn
from text_to_speech.modules.commons.layers import Embedding
from text_to_speech.modules.commons.nar_tts_modules import EnergyPredictor, PitchPredictor
from text_to_speech.modules.tts.commons.align_ops import expand_states
from text_to_speech.modules.tts.fs import FastSpeech
from text_to_speech.utils.audio.cwt import cwt2f0, get_lf0_cwt
from text_to_speech.utils.audio.pitch.utils import denorm_f0, f0_to_coarse, norm_f0
import numpy as np


class FastSpeech2Orig(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(300, self.hidden_size, 0)
            self.energy_predictor = EnergyPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=2,
                kernel_size=hparams['predictor_kernel'])
        if hparams['pitch_type'] == 'cwt' and hparams['use_pitch_embed']:
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=11,
                kernel_size=hparams['predictor_kernel'])
            self.cwt_stats_layers = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 2))

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None,
                f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)

        # add dur
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)

        # add pitch and energy embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)

        # add pitch and energy embed
        if self.hparams['use_energy_embed']:
            energy_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret)

        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        if self.hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels])).to(decoder_inp.device)
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'cwt':
            decoder_inp = decoder_inp.detach() + self.hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
            pitch_padding = mel2ph == 0
            ret['cwt'] = cwt_out = self.pitch_predictor(decoder_inp)
            stats_out = self.cwt_stats_layers(decoder_inp.mean(1))  # [B, 2]
            mean = ret['f0_mean'] = stats_out[:, 0]
            std = ret['f0_std'] = stats_out[:, 1]
            cwt_spec = cwt_out[:, :, :10]
            if f0 is None:
                std = std * self.hparams['cwt_std_scale']
                f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
                if self.hparams['use_uv']:
                    assert cwt_out.shape[-1] == 11
                    uv = cwt_out[:, :, -1] > 0
            ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv if self.hparams['use_uv'] else None,
                                                     pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)  # start from 0
            pitch_embed = self.pitch_embed(pitch)
            return pitch_embed
        else:
            return super(FastSpeech2Orig, self).forward_pitch(decoder_inp, f0, uv, mel2ph, ret, encoder_out)

    def forward_energy(self, decoder_inp, energy, ret):
        decoder_inp = decoder_inp.detach() + self.hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        energy_embed_inp = energy_pred if energy is None else energy
        energy_embed_inp = torch.clamp(energy_embed_inp * 256 // 4, min=0, max=255).long()
        energy_embed = self.energy_embed(energy_embed_inp)
        return energy_embed

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        _, cwt_scales = get_lf0_cwt(np.ones(10))
        f0 = cwt2f0(cwt_spec, mean, std, cwt_scales)
        f0 = torch.cat(
            [f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None)
        return f0_norm
