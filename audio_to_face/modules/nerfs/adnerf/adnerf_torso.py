import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_to_face.modules.nerfs.commons.embedders import FreqEmbedder
from audio_to_face.modules.nerfs.adnerf.backbone import NeRFBackbone, AudioNet, AudioAttNet


class ADNeRFTorso(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.pos_embedder = FreqEmbedder(in_dim=3, multi_res=10, use_log_bands=True, include_input=True)
        self.view_embedder = FreqEmbedder(in_dim=3, multi_res=4, use_log_bands=True, include_input=True)
        self.euler_embedder = FreqEmbedder(in_dim=3, multi_res=6, use_log_bands=True, include_input=True)
        self.trans_embedder = FreqEmbedder(in_dim=3, multi_res=6, use_log_bands=True, include_input=True)

        pos_dim = self.pos_embedder.out_dim
        view_dim = self.view_embedder.out_dim
        nerf_in_cond_dim = hparams['cond_dim'] + self.euler_embedder.out_dim + self.trans_embedder.out_dim
        
        if hparams.get("use_color", False):
            # pixel-level head color condition to prevent head-torso-separation artifacts
            color_cond_dim = 16
            self.color_encoder = nn.Sequential(*[
                nn.Linear(3, 16, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(16, 32, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(32, color_cond_dim, bias=True),
            ])
            nerf_in_cond_dim += color_cond_dim
        audnet_out_dim = hparams['cond_dim']

        self.model_coarse = NeRFBackbone(pos_dim=pos_dim, cond_dim=nerf_in_cond_dim, view_dim=view_dim, hid_dim=hparams['hidden_size'], num_density_linears=8, num_color_linears=3, skip_layer_indices=[4])
        self.model_fine = NeRFBackbone(pos_dim=pos_dim, cond_dim=nerf_in_cond_dim, view_dim=view_dim, hid_dim=hparams['hidden_size'], num_density_linears=8, num_color_linears=3, skip_layer_indices=[4])
        
        self.deepspeech_win_size = 16
        self.smo_win_size = 8
        self.aud_net = AudioNet(in_dim=29, out_dim=audnet_out_dim, win_size=self.deepspeech_win_size)
        self.audatt_net = AudioAttNet(in_out_dim=audnet_out_dim, seq_len=self.smo_win_size)

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

    def cal_cond_feat(self, cond, with_att=False,  **kwargs):
        cond_feat = self.aud_net(cond)
        if with_att:
            cond_feat = self.audatt_net(cond_feat)
        if cond_feat.ndim == 1:
            cond_feat = cond_feat.unsqueeze(0)
        euler_embedding = self.euler_embedder(kwargs['euler']).unsqueeze(0).repeat([cond_feat.shape[0],1])
        trans_embedding = self.trans_embedder(kwargs['trans']).unsqueeze(0).repeat([cond_feat.shape[0],1])
        cond_feat = torch.cat([cond_feat, euler_embedding, trans_embedding], dim=-1)

        if self.hparams.get("use_color", False):
            color = kwargs['color']
            color_feat = self.color_encoder(color)
            cond_feat = cond_feat.reshape([1, -1])
            cond_feat = cond_feat.repeat([color_feat.shape[0], 1])
            cond_feat = torch.cat([cond_feat, color_feat], dim=-1)
            
        return cond_feat


    