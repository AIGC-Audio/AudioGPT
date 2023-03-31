import torch
from modules.GenerSpeech.model.glow_modules import Glow
from modules.fastspeech.tts_modules import PitchPredictor
import random
from modules.GenerSpeech.model.prosody_util import ProsodyAligner, LocalStyleAdaptor
from utils.pitch_utils import f0_to_coarse, denorm_f0
from modules.commons.common_layers import *
import torch.distributions as dist
from utils.hparams import hparams
from modules.GenerSpeech.model.mixstyle import MixStyle
from modules.fastspeech.fs2 import FastSpeech2
import json
from modules.fastspeech.tts_modules import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS

class GenerSpeech(FastSpeech2):
    '''
    GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech
    https://arxiv.org/abs/2205.07211
    '''
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)

        # Mixstyle
        self.norm = MixStyle(p=0.5, alpha=0.1, eps=1e-6, hidden_size=self.hidden_size)

        # emotion embedding
        self.emo_embed_proj = Linear(256, self.hidden_size, bias=True)

        # build prosody extractor
        ## frame level
        self.prosody_extractor_utter = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx)
        self.l1_utter = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.align_utter = ProsodyAligner(num_layers=2)

        ## phoneme level
        self.prosody_extractor_ph = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx)
        self.l1_ph = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.align_ph = ProsodyAligner(num_layers=2)

        ## word level
        self.prosody_extractor_word = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx)
        self.l1_word = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.align_word = ProsodyAligner(num_layers=2)

        self.pitch_inpainter_predictor = PitchPredictor(
            self.hidden_size, n_chans=self.hidden_size,
            n_layers=3, dropout_rate=0.1, odim=2,
            padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])

        # build attention layer
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )

        # build post flow
        cond_hs = 80
        if hparams.get('use_txt_cond', True):
            cond_hs = cond_hs + hparams['hidden_size']

        cond_hs = cond_hs + hparams['hidden_size'] * 3  # for emo, spk embedding and prosody embedding
        self.post_flow = Glow(
            80, hparams['post_glow_hidden'], hparams['post_glow_kernel_size'], 1,
            hparams['post_glow_n_blocks'], hparams['post_glow_n_block_layers'],
            n_split=4, n_sqz=2,
            gin_channels=cond_hs,
            share_cond_layers=hparams['post_share_cond_layers'],
            share_wn_layers=hparams['share_wn_layers'],
            sigmoid_scale=hparams['sigmoid_scale']
        )
        self.prior_dist = dist.Normal(0, 1)


    def forward(self, txt_tokens, mel2ph=None, ref_mel2ph=None, ref_mel2word=None, spk_embed=None, emo_embed=None, ref_mels=None,
                f0=None, uv=None, skip_decoder=False, global_steps=0, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add spk/emo embed
        spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        emo_embed = self.emo_embed_proj(emo_embed)[:, None, :]


        # add dur
        dur_inp = (encoder_out + spk_embed + emo_embed) * src_nonpadding
        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = self.expand_states(encoder_out, mel2ph)
        decoder_inp = self.norm(decoder_inp, spk_embed + emo_embed)

        # add prosody VQ
        ret['ref_mel2ph'] = ref_mel2ph
        ret['ref_mel2word'] = ref_mel2word
        prosody_utter_mel = self.get_prosody_utter(decoder_inp, ref_mels, ret, infer, global_steps)
        prosody_ph_mel = self.get_prosody_ph(decoder_inp, ref_mels, ret, infer, global_steps)
        prosody_word_mel = self.get_prosody_word(decoder_inp, ref_mels, ret, infer, global_steps)

        # add pitch embed
        pitch_inp_domain_agnostic = decoder_inp * tgt_nonpadding
        pitch_inp_domain_specific = (decoder_inp + spk_embed + emo_embed + prosody_utter_mel + prosody_ph_mel + prosody_word_mel) * tgt_nonpadding
        predicted_pitch = self.inpaint_pitch(pitch_inp_domain_agnostic, pitch_inp_domain_specific, f0, uv, mel2ph, ret)

        # decode
        decoder_inp = decoder_inp + spk_embed + emo_embed + predicted_pitch + prosody_utter_mel + prosody_ph_mel + prosody_word_mel
        ret['decoder_inp'] = decoder_inp = decoder_inp * tgt_nonpadding
        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        # postflow
        is_training = self.training
        ret['x_mask'] = tgt_nonpadding
        ret['spk_embed'] = spk_embed
        ret['emo_embed'] = emo_embed
        ret['ref_prosody'] = prosody_utter_mel + prosody_ph_mel + prosody_word_mel
        self.run_post_glow(ref_mels, infer, is_training, ret)
        return ret

    def get_prosody_ph(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
        # get VQ prosody
        if global_steps > hparams['vq_start'] or infer:
            prosody_embedding, loss, ppl = self.prosody_extractor_ph(ref_mels, ret['ref_mel2ph'], no_vq=False)
            ret['vq_loss_ph'] = loss
            ret['ppl_ph'] = ppl
        else:
            prosody_embedding = self.prosody_extractor_ph(ref_mels, ret['ref_mel2ph'], no_vq=True)

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_ph(torch.cat([prosody_embedding, positions], dim=-1))


        # style-to-content attention
        src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
        prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
        if global_steps < hparams['forcing']:
            output, guided_loss, attn_emo = self.align_ph(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
                                                   src_key_padding_mask, prosody_key_padding_mask, forcing=True)
        else:
            output, guided_loss, attn_emo = self.align_ph(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
                                                       src_key_padding_mask, prosody_key_padding_mask, forcing=False)

        ret['gloss_ph'] = guided_loss
        ret['attn_ph'] = attn_emo
        return output.transpose(0, 1)

    def get_prosody_word(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
        # get VQ prosody
        if global_steps > hparams['vq_start'] or infer:
            prosody_embedding, loss, ppl = self.prosody_extractor_word(ref_mels, ret['ref_mel2word'], no_vq=False)
            ret['vq_loss_word'] = loss
            ret['ppl_word'] = ppl
        else:
            prosody_embedding = self.prosody_extractor_word(ref_mels, ret['ref_mel2word'], no_vq=True)

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_word(torch.cat([prosody_embedding, positions], dim=-1))


        # style-to-content attention
        src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
        prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
        if global_steps < hparams['forcing']:
            output, guided_loss, attn_emo = self.align_word(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
                                                   src_key_padding_mask, prosody_key_padding_mask, forcing=True)
        else:
            output, guided_loss, attn_emo = self.align_word(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
                                                       src_key_padding_mask, prosody_key_padding_mask, forcing=False)
        ret['gloss_word'] = guided_loss
        ret['attn_word'] = attn_emo
        return output.transpose(0, 1)

    def get_prosody_utter(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
        # get VQ prosody
        if global_steps > hparams['vq_start'] or infer:
            prosody_embedding, loss, ppl = self.prosody_extractor_utter(ref_mels, no_vq=False)
            ret['vq_loss_utter'] = loss
            ret['ppl_utter'] = ppl
        else:
            prosody_embedding = self.prosody_extractor_utter(ref_mels, no_vq=True)

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_utter(torch.cat([prosody_embedding, positions], dim=-1))


        # style-to-content attention
        src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
        prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
        if global_steps < hparams['forcing']:
            output, guided_loss, attn_emo = self.align_utter(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
                                                   src_key_padding_mask, prosody_key_padding_mask, forcing=True)
        else:
            output, guided_loss, attn_emo = self.align_utter(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
                                                       src_key_padding_mask, prosody_key_padding_mask, forcing=False)
        ret['gloss_utter'] = guided_loss
        ret['attn_utter'] = attn_emo
        return output.transpose(0, 1)



    def inpaint_pitch(self, pitch_inp_domain_agnostic, pitch_inp_domain_specific, f0, uv, mel2ph, ret):
        if hparams['pitch_type'] == 'frame':
            pitch_padding = mel2ph == 0
        if hparams['predictor_grad'] != 1:
            pitch_inp_domain_agnostic = pitch_inp_domain_agnostic.detach() + hparams['predictor_grad'] * (pitch_inp_domain_agnostic - pitch_inp_domain_agnostic.detach())
            pitch_inp_domain_specific = pitch_inp_domain_specific.detach() + hparams['predictor_grad'] * (pitch_inp_domain_specific - pitch_inp_domain_specific.detach())

        pitch_domain_agnostic = self.pitch_predictor(pitch_inp_domain_agnostic)
        pitch_domain_specific = self.pitch_inpainter_predictor(pitch_inp_domain_specific)
        pitch_pred = pitch_domain_agnostic + pitch_domain_specific
        ret['pitch_pred'] = pitch_pred

        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]  # [B, T]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0  # [B, T]
        f0_denorm = denorm_f0(f0, uv if use_uv else None, hparams, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None, hparams, pitch_padding=pitch_padding)
        if hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def run_post_glow(self, tgt_mels, infer, is_training, ret):
        x_recon = ret['mel_out'].transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        if hparams.get('use_txt_cond', True):
            g = torch.cat([g, ret['decoder_inp'].transpose(1, 2)], 1)
        g_spk_embed = ret['spk_embed'].repeat(1, T, 1).transpose(1, 2)
        g_emo_embed = ret['emo_embed'].repeat(1, T, 1).transpose(1, 2)
        l_ref_prosody = ret['ref_prosody'].transpose(1, 2)
        g = torch.cat([g, g_spk_embed, g_emo_embed, l_ref_prosody], dim=1)
        prior_dist = self.prior_dist
        if not infer:
            if is_training:
                self.train()
            x_mask = ret['x_mask'].transpose(1, 2)
            y_lengths = x_mask.sum(-1)
            g = g.detach()
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self.post_flow(tgt_mels, x_mask, g=g)
            ldj = ldj / y_lengths / 80
            ret['z_pf'], ret['ldj_pf'] = z_postflow, ldj
            ret['postflow'] = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
        else:
            x_mask = torch.ones_like(x_recon[:, :1, :])
            z_post = prior_dist.sample(x_recon.shape).to(g.device) * hparams['noise_scale']
            x_recon_, _ = self.post_flow(z_post, x_mask, g, reverse=True)
            x_recon = x_recon_
            ret['mel_out'] = x_recon.transpose(1, 2)