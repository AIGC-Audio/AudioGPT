import torch
import torch.nn as nn
import numpy as np
import tqdm
import time
import imageio
import os
import cv2

from audio_to_face.utils.commons.hparams import hparams
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.nn.model_utils import print_arch
from audio_to_face.utils.nn.grad import get_grad_norm, GradBuffer
from audio_to_face.utils.nn.model_utils import print_arch, get_device_of_model, not_requires_grad
from audio_to_face.utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda, convert_to_tensor

from audio_to_face.tasks.nerfs.adnerf_torso import ADNeRFTorsoTask
from audio_to_face.modules.nerfs.lm3d_nerf.lm3d_nerf import Lm3dNeRF as Lm3dNeRF_head
from audio_to_face.modules.nerfs.adnerf.adnerf import ADNeRF as ADNeRF_head
from audio_to_face.modules.nerfs.adnerf.adnerf_torso import ADNeRFTorso as ADNeRF_torso
from audio_to_face.modules.nerfs.commons.volume_rendering import render_dynamic_face
from audio_to_face.modules.nerfs.commons.ray_samplers import TorsoUniformRaySampler
from scipy.ndimage import gaussian_filter1d, gaussian_filter


class Lm3dNeRFTorsoTask(ADNeRFTorsoTask):

    def build_model(self):
        self.head_model = Lm3dNeRF_head(hparams)
        head_model_dir = hparams['head_model_dir']
        load_ckpt(self.head_model, head_model_dir)

        self.model = ADNeRF_torso(hparams)
        self.audatt_net_params = [p for p in self.model.audatt_net.parameters() if p.requires_grad]
        self.gen_params_except_audatt_net = [p for k, p in self.model.named_parameters() if (('audatt_net' not in k) and p.requires_grad)]        
        return self.model


    def run_model(self, sample, infer=False, run_head_mode=False):
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
        c2w_t = sample['c2w'] 
        c2w_t0 = sample['c2w_t0']
        euler = sample['euler']
        euler_t0 = sample['euler_t0']
        trans = sample['trans']
        trans_t0 = sample['trans_t0']
        losses_out = {}

        with_att = hparams['with_att']

        if infer:
            # Inference Phase
            with torch.no_grad():
                # render head
                # sample the rays of the whole image
                rays_o, rays_d, _ = self.full_rays_sampler(H, W, focal, c2w_t)

                if with_att:
                    cond_feat = self.head_model.cal_cond_feat(cond_wins, with_att=True)
                else:
                    cond_feat = self.head_model.cal_cond_feat(cond, with_att=False)
                rgb_pred, disp, acc, last_weight, rgb_map_fg, extras = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                    bc_rgb=bg_img, chunk=2048, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.head_model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=True,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    )

                # render torso
                # sample the rays of the whole image, in the canoical space, i.e., with c2w_t0
                rays_o, rays_d, _ = self.full_rays_sampler(H, W, focal, c2w_t0)

                cond_feat = self.model.cal_cond_feat(sample['deepspeech_wins'],color=rgb_pred, euler=sample['euler'], trans=sample['trans'],with_att=True)
                rgb_pred_torso, disp_torso, acc_torso, last_weight_torso, rgb_map_fg_torso, extras_torso = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                    bc_rgb=bg_img,chunk=2048, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=False,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    euler=euler, euler_t0=euler_t0, trans=trans, trans_t0=trans_t0
                    )

                if hparams.get("infer_with_more_dynamic_c2w_sequence", False) is True:
                    """
                    Note: enable it only when you find there is overlap problem between head and torso!
                    Since the torso nerf is modeled in canoical space (i.e., static pose), 
                    it cannot model large-range movements of the torso.
                    When the head is moving extremely down,
                    the torso nerf sometimes will render results on the face part,
                    which leads to shallow artifacts on the face part.
                    To handle this, we set the rgb_map_fg_torso on the face part to zero.
                    """
                    w_h = int(last_weight.reshape([-1]).shape[0]**0.5)
                    smo_last_weight = gaussian_filter((last_weight.reshape([w_h,w_h]).cpu() * 255).int().numpy(), sigma=1.).reshape([-1]) / 255.
                    has_head_mask = convert_to_tensor(smo_last_weight <= 0.3).to(rgb_map_fg.device).bool() # where head has much confidence
                    def shrink_has_head_mask(has_head_mask):
                        w_h = int(has_head_mask.reshape([-1]).shape[0]**0.5)
                        has_head_mask = has_head_mask.reshape([w_h, w_h])
                        centered_mask = has_head_mask[1:-1,1:-1]
                        left_offset_mask = has_head_mask[0:-2,1:-1]
                        right_offset_mask = has_head_mask[2:,1:-1]
                        up_offset_mask = has_head_mask[1:-1,0:-2]
                        down_offset_mask = has_head_mask[1:-1,2:]
                        mask = torch.bitwise_and(centered_mask, left_offset_mask)
                        mask = torch.bitwise_and(mask, right_offset_mask)
                        mask = torch.bitwise_and(mask, up_offset_mask)
                        mask = torch.bitwise_and(mask, down_offset_mask)
                        has_head_mask[1:-1,1:-1] = mask
                        return has_head_mask.reshape([-1,])
                    for _ in range(6):
                        has_head_mask = shrink_has_head_mask(has_head_mask)      
                    disable_torso_mask = has_head_mask   
                    last_weight_torso[disable_torso_mask] = 1   
                    rgb_map_fg_torso[last_weight_torso==1] = 0
                rgb_com = rgb_pred * last_weight_torso.unsqueeze(-1) + rgb_map_fg_torso 

                model_out = {
                    "rgb_map" : rgb_com
                }
                return model_out
        else:
            # Training Phase
            if run_head_mode:
                # Run Head NeRF
                # uniformly sample the rays
                rays_o, rays_d, select_coords = self.rays_sampler(H, W, focal, c2w_t, n_rays=None, rect=sample['rect'], in_rect_percent=hparams['in_rect_percent'], iterations=self.global_step)
                rgb_gt = self.rays_sampler.sample_pixels_from_img_with_select_coords(sample['head_img'], select_coords)
                rgb_bc = self.rays_sampler.sample_pixels_from_img_with_select_coords(sample['bg_img'], select_coords)
                with torch.no_grad():
                    # calculate the condition
                    if with_att:
                        cond_feat = self.head_model.cal_cond_feat(cond_wins, with_att=True)
                    else:
                        cond_feat = self.head_model.cal_cond_feat(cond, with_att=False)
                    # volume rendering to get rgb_pred
                    rgb_pred, disp, acc, last_weight, rgb_map_fg, extras = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                        bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                        network_fn=self.head_model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                        run_head_mode=True,
                        c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                        )
                    # calculate loss
                    losses_out['head_mse_loss'] = torch.mean((rgb_pred - rgb_gt) ** 2)
                    if 'rgb_map_coarse' in extras:
                        losses_out['head_mse_loss_coarse'] = torch.mean((extras['rgb_map_coarse'] - rgb_gt) ** 2)
                    model_out = {
                        "rgb_map": rgb_pred
                    }
            else:
                # Run Torso NeRF
                # uniformly sample the rays, in the canoical space, i.e., with c2w_t0
                target = sample['gt_img']
                rect = [0, H/2, W, H/2] # only sample the lower part for torso
                rays_o, rays_d, select_coords = self.torso_rays_sampler(H, W, focal, c2w_t0, n_rays=None, rect=rect, 
                            in_rect_percent=hparams['in_rect_percent'])
                rays_o_head, rays_d_head, _ = self.rays_sampler(H, W, focal, c2w_t, select_coords=select_coords)
                rgb_gt = self.rays_sampler.sample_pixels_from_img_with_select_coords(target, select_coords)
                rgb_bc = self.rays_sampler.sample_pixels_from_img_with_select_coords(sample['bg_img'], select_coords)

                # render head
                with torch.no_grad():
                    if with_att:
                        cond_feat = self.head_model.cal_cond_feat(cond_wins, with_att=True)
                    else:
                        cond_feat = self.head_model.cal_cond_feat(cond, with_att=False)
                    rgb_pred_head, disp_head, acc_head, last_weight_head, rgb_map_fg_head, extras_head = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o_head, rays_d=rays_d_head,
                    bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.head_model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=True,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    )

                # render torso
                # calculate the condition based on deepspeech
                cond_feat = self.model.cal_cond_feat(sample['deepspeech_wins'],color=rgb_pred_head, euler=sample['euler'], trans=sample['trans'],with_att=True)
                # volume rendering to get rgb_pred_com
                rgb_pred_torso, disp_torso, acc_torso, last_weight_torso, rgb_map_fg_torso, extras_torso = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                    bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=run_head_mode,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    euler=euler, euler_t0=euler_t0, trans=trans, trans_t0=trans_t0
                    )
                rgb_com = rgb_pred_head * last_weight_torso.unsqueeze(-1) + rgb_map_fg_torso
                
                # calculate loss
                losses_out['com_mse_loss'] = torch.mean((rgb_com - rgb_gt) ** 2)
                def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)
                losses_out['com_psnr'] = mse2psnr(losses_out['com_mse_loss'].detach())
                if 'rgb_map_coarse' in extras_torso:
                    rgb_com0 = extras_head['rgb_map_coarse'] * extras_torso['last_weight0'].unsqueeze(-1) + extras_torso['rgb_map_fg0']
                    losses_out['com_mse_loss_coarse']  = torch.mean((rgb_com0 - rgb_gt) ** 2) 
                model_out = {
                    "rgb_map": rgb_com
                }
            return losses_out, model_out
    