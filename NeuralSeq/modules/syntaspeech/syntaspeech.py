import math
import torch
from torch import nn
from torch.nn import Linear
from utils.hparams import hparams
from modules.commons.conv import ConvBlocks, ConditionalConvBlocks
from modules.commons.common_layers import Embedding
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.transformer import MultiheadAttention, FFTBlocks
from modules.commons.align_ops import clip_mel2token_to_multiple, build_word_mask, expand_states, mel2ph_to_mel2word
from modules.tts.fastspeech import FS_DECODERS, FastSpeech
from modules.portaspeech.fvae import SyntaFVAE, FVAE
from utils.nn.seq_utils import group_hidden_by_segs
from modules.fastspeech.tts_modules import SyntaDurationPredictor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        """

        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SyntaSpeech(FastSpeech):
    def __init__(self, ph_dict_size, word_dict_size, out_dims=None):
        super().__init__(ph_dict_size, out_dims)
        # build linguistic encoder
        if hparams['num_spk'] > 1:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_word_encoder']:
            self.word_encoder = RelTransformerEncoder(
                word_dict_size, self.hidden_size, self.hidden_size, self.hidden_size, 2,
                hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
        if hparams['dur_level'] == 'word':
            if hparams['word_encoder_type'] == 'rel_fft':
                self.ph2word_encoder = RelTransformerEncoder(
                    0, self.hidden_size, self.hidden_size, self.hidden_size, 2,
                    hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
            if hparams['word_encoder_type'] == 'fft':
                self.ph2word_encoder = FFTBlocks(
                    self.hidden_size, hparams['word_enc_layers'], 1, num_heads=hparams['num_heads'])
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
            self.enc_pos_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_query_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_res_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
            self.attn.enable_torch_version = False
            if hparams['text_encoder_postnet']:
                self.text_encoder_postnet = ConvBlocks(
                    self.hidden_size, self.hidden_size, [1] * 3, 5, layers_in_block=2)
        else:
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = SyntaDurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        # build VAE decoder
        if hparams['use_fvae']:
            del self.decoder
            del self.mel_out
            if hparams.get("use_gae_in_prior", True):
                self.fvae = SyntaFVAE(
                    c_in_out=self.out_dims,
                    hidden_size=hparams['fvae_enc_dec_hidden'], c_latent=hparams['latent_size'],
                    kernel_size=hparams['fvae_kernel_size'],
                    enc_n_layers=hparams['fvae_enc_n_layers'],
                    dec_n_layers=hparams['fvae_dec_n_layers'],
                    c_cond=self.hidden_size,
                    use_prior_flow=hparams['use_prior_flow'],
                    flow_hidden=hparams['prior_flow_hidden'],
                    flow_kernel_size=hparams['prior_flow_kernel_size'],
                    flow_n_steps=hparams['prior_flow_n_blocks'],
                    strides=[hparams['fvae_strides']],
                    encoder_type=hparams['fvae_encoder_type'],
                    decoder_type=hparams['fvae_decoder_type'],
                )
            else:
                self.fvae = FVAE(
                    c_in_out=self.out_dims,
                    hidden_size=hparams['fvae_enc_dec_hidden'], c_latent=hparams['latent_size'],
                    kernel_size=hparams['fvae_kernel_size'],
                    enc_n_layers=hparams['fvae_enc_n_layers'],
                    dec_n_layers=hparams['fvae_dec_n_layers'],
                    c_cond=self.hidden_size,
                    use_prior_flow=hparams['use_prior_flow'],
                    flow_hidden=hparams['prior_flow_hidden'],
                    flow_kernel_size=hparams['prior_flow_kernel_size'],
                    flow_n_steps=hparams['prior_flow_n_blocks'],
                    strides=[hparams['fvae_strides']],
                    encoder_type=hparams['fvae_encoder_type'],
                    decoder_type=hparams['fvae_decoder_type'],
                )
        else:
            self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
            self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
        if hparams['add_word_pos']:
            self.word_pos_proj = Linear(self.hidden_size, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, word_tokens, ph2word, word_len, mel2word=None, mel2ph=None,
                spk_embed=None, spk_id=None, pitch=None, infer=False, tgt_mels=None,
                global_step=None, graph_lst=None, etypes_lst=None, *args, **kwargs):     

        if hparams['use_spk_embed']:
            spk_embed = spk_embed
        elif hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_id)[:, None, :]  
        else:
            spk_embed = 0

        ret = {}
        style_embed = self.forward_style_embed(spk_embed, spk_id) # speaker embedding, [B, 1, C]
        x, tgt_nonpadding = self.run_text_encoder(
            txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, style_embed, ret, graph_lst=graph_lst, etypes_lst=etypes_lst, **kwargs)
        x = x + style_embed # it maybe necessary to achieve multi-speaker
        x = x * tgt_nonpadding
        ret['nonpadding'] = tgt_nonpadding
        if hparams['use_pitch_embed']:
            x = x + self.pitch_embed(pitch)
        ret['decoder_inp'] = x
        if infer and (mel2ph is None or mel2word is None):
            mel2word = ret['mel2word']
        ret['mel_out_fvae'] = ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, ret, infer, tgt_mels, global_step,
                mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
        return ret

    def run_text_encoder(self, txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, style_embed, ret, graph_lst, etypes_lst, **kwargs):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        use_bert = hparams.get("use_bert") is True
        if use_bert:
            ph_encoder_out = self.encoder(txt_tokens, bert_feats=kwargs['bert_feats'], ph2word=ph2word,
                graph_lst=graph_lst, etypes_lst=etypes_lst,
                cl_feats=kwargs['cl_feats'], ret=ret) * src_nonpadding + style_embed
        else:
            ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding + style_embed
        if hparams['use_word_encoder']:
            word_encoder_out = self.word_encoder(word_tokens) + style_embed
            ph_encoder_out = ph_encoder_out + expand_states(word_encoder_out, ph2word)
        
        dur_input = ph_encoder_out * src_nonpadding
        if hparams['dur_level'] == 'word':
            word_encoder_out = 0
            h_ph_gb_word = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)[0]
            word_encoder_out = word_encoder_out + self.ph2word_encoder(h_ph_gb_word)
            if hparams['use_word_encoder']:
                word_encoder_out = word_encoder_out + self.word_encoder(word_tokens)
            mel2word = self.forward_dur(dur_input, mel2word, ret, ph2word=ph2word, word_len=word_len, graph_lst=graph_lst, etypes_lst=etypes_lst)
            mel2word = clip_mel2token_to_multiple(mel2word, hparams['frames_multiple'])
            ret['mel2word'] = mel2word
            tgt_nonpadding = (mel2word > 0).float()[:, :, None]
            enc_pos = self.get_pos_embed(word2word, ph2word)  # [B, T_ph, H]
            dec_pos = self.get_pos_embed(word2word, mel2word)  # [B, T_mel, H]
            dec_word_mask = build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
            x, weight = self.attention(ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask)
            if hparams['add_word_pos']:
                x = x + self.word_pos_proj(dec_pos)
            ret['attn'] = weight
        else:
            mel2ph = self.forward_dur(dur_input, mel2ph, ret)
            mel2ph = clip_mel2token_to_multiple(mel2ph, hparams['frames_multiple'])
            mel2word = mel2ph_to_mel2word(mel2ph, ph2word)
            x = expand_states(ph_encoder_out, mel2ph)
            if hparams['add_word_pos']:
                dec_pos = self.get_pos_embed(word2word, mel2word)  # [B, T_mel, H]
                x = x + self.word_pos_proj(dec_pos)
            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        if hparams['use_word_encoder']:
            x = x + expand_states(word_encoder_out, mel2word)
        return x, tgt_nonpadding

    def attention(self, ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask):
        ph_kv = self.enc_pos_proj(torch.cat([ph_encoder_out, enc_pos], -1))
        word_enc_out_expend = expand_states(word_encoder_out, mel2word)
        word_enc_out_expend = torch.cat([word_enc_out_expend, dec_pos], -1)
        if hparams['text_encoder_postnet']:
            word_enc_out_expend = self.dec_res_proj(word_enc_out_expend)
            word_enc_out_expend = self.text_encoder_postnet(word_enc_out_expend)
            dec_q = x_res = word_enc_out_expend
        else:
            dec_q = self.dec_query_proj(word_enc_out_expend)
            x_res = self.dec_res_proj(word_enc_out_expend)
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1e9)
        x = x.transpose(0, 1)
        x = x + x_res
        return x, weight

    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None, global_step=0,
                    mel2word=None, ph2word=None, graph_lst=None, etypes_lst=None):
        if not hparams['use_fvae']:
            x = self.decoder(x)
            x = self.mel_out(x)
            ret['kl'] = 0
            return x * tgt_nonpadding
        else:
            # x is the phoneme encoding
            x = x.transpose(1, 2)  # [B, H, T]
            tgt_nonpadding_BHT = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
            if infer:
                z = self.fvae(cond=x, infer=True, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
            else:
                tgt_mels = tgt_mels.transpose(1, 2)  # [B, 80, T]
                z, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = self.fvae(
                    tgt_mels, tgt_nonpadding_BHT, cond=x, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
                if global_step < hparams['posterior_start_steps']:
                    z = torch.randn_like(z)
            x_recon = self.fvae.decoder(z, nonpadding=tgt_nonpadding_BHT, cond=x).transpose(1, 2)
            ret['pre_mel_out'] = x_recon
            return x_recon

    def forward_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        word_len = kwargs['word_len']
        ph2word = kwargs['ph2word']
        graph_lst = kwargs['graph_lst']
        etypes_lst = kwargs['etypes_lst']
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding, ph2word, graph_lst, etypes_lst)

        B, T_ph = ph2word.shape
        dur = torch.zeros([B, word_len.max() + 1]).to(ph2word.device).scatter_add(1, ph2word, dur)
        dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            mel2word = self.length_regulator(dur).detach()
        return mel2word

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
