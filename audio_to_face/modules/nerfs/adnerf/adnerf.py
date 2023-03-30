import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_to_face.modules.nerfs.commons.embedders import FreqEmbedder
from audio_to_face.modules.nerfs.adnerf.backbone import NeRFBackbone, AudioNet, AudioAttNet


class ADNeRF(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.pos_embedder = FreqEmbedder(in_dim=3, multi_res=10, use_log_bands=True, include_input=True)
        self.view_embedder = FreqEmbedder(in_dim=3, multi_res=4, use_log_bands=True, include_input=True)
        pos_dim = self.pos_embedder.out_dim
        view_dim = self.view_embedder.out_dim
        self.cond_dim = hparams['cond_dim']
        self.model_coarse = NeRFBackbone(pos_dim=pos_dim, cond_dim=self.cond_dim, view_dim=view_dim, hid_dim=hparams['hidden_size'], num_density_linears=8, num_color_linears=3, skip_layer_indices=[4])
        self.model_fine = NeRFBackbone(pos_dim=pos_dim, cond_dim=self.cond_dim, view_dim=view_dim, hid_dim=hparams['hidden_size'], num_density_linears=8, num_color_linears=3, skip_layer_indices=[4])
        
        self.deepspeech_win_size = 16
        self.smo_win_size = 8
        self.aud_net = AudioNet(in_dim=29, out_dim=self.cond_dim, win_size=self.deepspeech_win_size)
        self.audatt_net = AudioAttNet(in_out_dim=self.cond_dim, seq_len=self.smo_win_size)

    def forward(self, pos, cond_feat, view, run_model_fine=True, **kwargs):
        out = {}
        pos_embed = self.pos_embedder(pos)
        view_embed = self.view_embedder(view)
        if run_model_fine:
            rgb_sigma = self.model_fine(pos_embed, cond_feat, view_embed)
        else:
            rgb_sigma = self.model_coarse(pos_embed, cond_feat, view_embed)
        out['rgb_sigma'] = rgb_sigma
        return out

    def cal_cond_feat(self, cond, with_att=False):
        cond_feat = self.aud_net(cond)
        if with_att:
            cond_feat = self.audatt_net(cond_feat)
        return cond_feat


    