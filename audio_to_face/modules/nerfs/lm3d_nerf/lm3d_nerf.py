import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_to_face.modules.nerfs.commons.embedders import FreqEmbedder
from audio_to_face.modules.nerfs.adnerf.backbone import NeRFBackbone
from audio_to_face.modules.nerfs.lm3d_nerf.cond_encoder import AudioNet, AudioAttNet
from audio_to_face.modules.nerfs.commons.volume_rendering import render_dynamic_face

from audio_to_face.utils.commons.hparams import hparams


class Lm3dNeRF(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.pos_embedder = FreqEmbedder(in_dim=3, multi_res=10, use_log_bands=True, include_input=True)
        self.view_embedder = FreqEmbedder(in_dim=3, multi_res=4, use_log_bands=True, include_input=True)
        pos_dim = self.pos_embedder.out_dim
        view_dim = self.view_embedder.out_dim
        nerf_cond_dim = lm3d_out_dim = hparams['cond_dim']
        self.model_coarse = NeRFBackbone(pos_dim=pos_dim, cond_dim=nerf_cond_dim, view_dim=view_dim, hid_dim=hparams['hidden_size'], num_density_linears=8, num_color_linears=3, skip_layer_indices=[4])
        self.model_fine = NeRFBackbone(pos_dim=pos_dim, cond_dim=nerf_cond_dim, view_dim=view_dim, hid_dim=hparams['hidden_size'], num_density_linears=8, num_color_linears=3, skip_layer_indices=[4])

        cond_in_dim = 68 * 3
        if hparams['use_window_cond']:
            self.lm3d_win_size = hparams['cond_win_size']
            self.smo_win_size = hparams['smo_win_size']
            self.lm_encoder = AudioNet(in_dim=cond_in_dim, out_dim=lm3d_out_dim, win_size=self.lm3d_win_size)
            if hparams['with_att']:
                self.lmatt_encoder = AudioAttNet(in_out_dim=lm3d_out_dim, seq_len=self.smo_win_size)
        else:
            self.lm_encoder = nn.Sequential(*[
                nn.Linear(cond_in_dim, 32, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(32, 32, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(32, 64, bias=True),
                nn.LeakyReLU(0.02, True),
                nn.Linear(64, lm3d_out_dim, bias=True),
            ])

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
        cond_feat = self.lm_encoder(cond)
        if with_att:
            cond_feat = self.lmatt_encoder(cond_feat)
        return cond_feat

    ##########################
    # forward the model
    ##########################
    def run_model(self, sample, infer=False):
        """
        render or train on a single-frame
        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return:
            if not infer:
                return losses, model_out
            if infer:
                return model_out
        """
        cond = sample['cond_win'] if hparams['use_window_cond'] else sample['cond']
        cond_wins = sample['cond_wins']
        H = sample['H']
        W = sample['W']
        focal = sample['focal']
        cx = sample['cx']
        cy = sample['cy']
        near = sample['near']
        far = sample['far']
        bg_img = sample['bg_img']
        c2w = sample['c2w'] 
        c2w_t0 = sample['c2w_t0']
        t = sample['t'] 
        
        with_att = hparams['with_att'] and (self.global_step >= self.no_smo_iterations)
        if with_att:
            cond_feat = self.model.cal_cond_feat(cond_wins, with_att=True)
        else:
            cond_feat = self.model.cal_cond_feat(cond, with_att=False)

        if infer:
            rays_o, rays_d, _ = self.full_rays_sampler(H, W, focal, c2w)
            rgb_pred, disp, acc, _, _,  extras = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                bc_rgb=bg_img, 
                chunk=2048,
                c2w=None, cond=cond_feat, near=near, far=far,
                network_fn=self.model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                c2w_t=c2w, c2w_t0=c2w_t0,t=t,
                )
            model_out = {
                "rgb_map" : rgb_pred
            }
            return model_out
        else:
            rays_o, rays_d, select_coords = self.rays_sampler(H, W, focal, c2w, n_rays=None, rect=sample['rect'], in_rect_percent=hparams['in_rect_percent'], iterations=self.global_step)
            target = sample['head_img']
            rgb_gt = self.rays_sampler.sample_pixels_from_img_with_select_coords(target, select_coords)
            rgb_bc = self.rays_sampler.sample_pixels_from_img_with_select_coords(sample['bg_img'], select_coords)

            rgb_pred, disp, acc, _, _, extras = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                network_fn=self.model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                c2w_t=c2w, c2w_t0=c2w_t0,t=t,)
            losses_out = {}
            losses_out['mse_loss'] = torch.mean((rgb_pred - rgb_gt) ** 2)
            if 'rgb_map_coarse' in extras:
                losses_out['mse_loss_coarse'] = torch.mean((extras['rgb_map_coarse'] - rgb_gt) ** 2)
            model_out = {
                "rgb_map": rgb_pred
            }
            return losses_out, model_out
    

    