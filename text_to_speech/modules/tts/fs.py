from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from text_to_speech.modules.commons.conv import TextConvEncoder, ConvBlocks
from text_to_speech.modules.commons.layers import Embedding
from text_to_speech.modules.commons.nar_tts_modules import PitchPredictor, DurationPredictor, LengthRegulator
from text_to_speech.modules.commons.rel_transformer import RelTransformerEncoder, BERTRelTransformerEncoder
from text_to_speech.modules.commons.rnn import TacotronEncoder, RNNEncoder, DecoderRNN
from text_to_speech.modules.commons.transformer import FastSpeechEncoder, FastSpeechDecoder
from text_to_speech.modules.commons.wavenet import WN
from text_to_speech.modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from text_to_speech.utils.audio.pitch.utils import denorm_f0, f0_to_coarse

FS_ENCODERS = {
    'fft': lambda hp, dict_size: FastSpeechEncoder(
        dict_size, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
    'tacotron': lambda hp, dict_size: TacotronEncoder(
        hp['hidden_size'], dict_size, hp['hidden_size'],
        K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']),
    'tacotron2': lambda hp, dict_size: RNNEncoder(dict_size, hp['hidden_size']),
    'conv': lambda hp, dict_size: TextConvEncoder(dict_size, hp['hidden_size'], hp['hidden_size'],
                                                  hp['enc_dilations'], hp['enc_kernel_size'],
                                                  layers_in_block=hp['layers_in_block'],
                                                  norm_type=hp['enc_dec_norm'],
                                                  post_net_kernel=hp.get('enc_post_net_kernel', 3)),
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}

FS_DECODERS = {
    'fft': lambda hp: FastSpeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']),
    'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations'],
                                  hp['dec_kernel_size'], layers_in_block=hp['layers_in_block'],
                                  norm_type=hp['enc_dec_norm'], dropout=hp['dropout'],
                                  post_net_kernel=hp.get('dec_post_net_kernel', 3)),
    'wn': lambda hp: WN(hp['hidden_size'], kernel_size=5, dilation_rate=1, n_layers=hp['dec_layers'],
                        is_BTC=True),
}


class FastSpeech(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        if hparams.get("use_bert") is True:
            self.ph_encoder = BERTRelTransformerEncoder(dict_size, hparams['hidden_size'], hparams['hidden_size'],
                hparams['ffn_hidden_size'], hparams['num_heads'], hparams['enc_layers'],
                hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        else:
            self.ph_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = hparams['audio_num_mel_bins'] if out_dims is None else out_dims
        self.mel_out = nn.Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_spk_id']:
            self.spk_id_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=5, dropout_rate=0.1, odim=2,
                kernel_size=hparams['predictor_kernel'])
        if hparams['dec_inp_add_noise']:
            self.z_channels = hparams['z_channels']
            self.dec_inp_noise_proj = nn.Linear(self.hidden_size + self.z_channels, self.hidden_size)

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None,
                f0=None, uv=None, infer=False, **kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)

        use_bert = self.hparams.get("use_bert") is True
        if use_bert:
            encoder_out = self.encoder(txt_tokens, bert_feats=kwargs['bert_feats'], ph2word=kwargs['ph2word'],
                ret=ret) * src_nonpadding + style_embed
        else:
            encoder_out = self.encoder(txt_tokens) * src_nonpadding + style_embed

        # add dur
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(encoder_out, mel2ph)

        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)

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

    def forward_style_embed(self, spk_embed=None, spk_id=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        return style_embed

    def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding
