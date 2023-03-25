import torch
import torch.distributions as dist
from torch import nn
from text_to_speech.modules.commons.normalizing_flow.glow_modules import Glow
from text_to_speech.modules.tts.portaspeech.portaspeech import PortaSpeech


class PortaSpeechFlow(PortaSpeech):
    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, word_dict_size, hparams, out_dims)
        cond_hs = 80
        if hparams.get('use_txt_cond', True):
            cond_hs = cond_hs + hparams['hidden_size']
        if hparams.get('use_latent_cond', False):
            cond_hs = cond_hs + hparams['latent_size']
        if hparams['use_cond_proj']:
            self.g_proj = nn.Conv1d(cond_hs, 160, 5, padding=2)
            cond_hs = 160
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

    def forward(self, txt_tokens, word_tokens, ph2word, word_len, mel2word=None, mel2ph=None,
                spk_embed=None, spk_id=None, pitch=None, infer=False, tgt_mels=None,
                forward_post_glow=True, two_stage=True, global_step=None, **kwargs):
        is_training = self.training
        train_fvae = not (forward_post_glow and two_stage)
        if not train_fvae:
            self.eval()
        with torch.set_grad_enabled(mode=train_fvae):
            ret = super(PortaSpeechFlow, self).forward(
                txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph,
                spk_embed, spk_id, pitch, infer, tgt_mels, global_step, **kwargs)
        if (forward_post_glow or not two_stage) and self.hparams['use_post_flow']:
            self.run_post_glow(tgt_mels, infer, is_training, ret)
        return ret

    def run_post_glow(self, tgt_mels, infer, is_training, ret):
        x_recon = ret['mel_out'].transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        if self.hparams.get('use_txt_cond', True):
            g = torch.cat([g, ret['decoder_inp'].transpose(1, 2)], 1)
        if self.hparams.get('use_latent_cond', False):
            g_z = ret['z_p'][:, :, :, None].repeat(1, 1, 1, 4).reshape(B, -1, T)
            g = torch.cat([g, g_z], 1)
        if self.hparams['use_cond_proj']:
            g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            if is_training:
                self.post_flow.train()
            nonpadding = ret['nonpadding'].transpose(1, 2)
            y_lengths = nonpadding.sum(-1)
            if self.hparams['detach_postflow_input']:
                g = g.detach()
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self.post_flow(tgt_mels, nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            ret['z_pf'], ret['ldj_pf'] = z_postflow, ldj
            ret['postflow'] = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            if torch.isnan(ret['postflow']):
                ret['postflow'] = None
        else:
            nonpadding = torch.ones_like(x_recon[:, :1, :])
            z_post = torch.randn(x_recon.shape).to(g.device) * self.hparams['noise_scale']
            x_recon, _ = self.post_flow(z_post, nonpadding, g, reverse=True)
            ret['mel_out'] = x_recon.transpose(1, 2)
