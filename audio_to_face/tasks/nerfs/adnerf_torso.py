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
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.nn.schedulers import ExponentialSchedule, ExponentialScheduleWithAudattNet

from audio_to_face.modules.nerfs.adnerf.adnerf import ADNeRF as ADNeRF_head
from audio_to_face.modules.nerfs.adnerf.adnerf_torso import ADNeRFTorso as ADNeRF_torso
from audio_to_face.modules.nerfs.commons.volume_rendering import render_dynamic_face
from audio_to_face.modules.nerfs.commons.ray_samplers import TorsoUniformRaySampler

from audio_to_face.tasks.nerfs.adnerf import ADNeRFTask


class ADNeRFTorsoTask(ADNeRFTask):
    def __init__(self):
        super().__init__()
        self.torso_rays_sampler = TorsoUniformRaySampler(self.n_rays)

    def build_model(self):
        self.head_model = ADNeRF_head(hparams)
        head_model_dir = hparams['head_model_dir']
        load_ckpt(self.head_model, head_model_dir)

        self.model = ADNeRF_torso(hparams)
        self.audatt_net_params = [p for p in self.model.audatt_net.parameters() if p.requires_grad]
        self.gen_params_except_audatt_net = [p for k, p in self.model.named_parameters() if (('audatt_net' not in k) and p.requires_grad)]        
        return self.model

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.gen_params_except_audatt_net,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        self.optimizer.add_param_group({
            'params': self.audatt_net_params,
            'lr': hparams['lr'] * 5,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2'])
        })
        return self.optimizer

    def build_scheduler(self, optimizer):
        return ExponentialScheduleWithAudattNet(optimizer, hparams['lr'], hparams['warmup_updates'])

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

        if infer:
            # Inference Phase
            with torch.no_grad():
                # render head
                cond_feat = self.head_model.cal_cond_feat(cond_wins, with_att=True)
                rays_o, rays_d, _ = self.full_rays_sampler(H, W, focal, c2w_t)
                rgb_pred, disp, acc, last_weight, rgb_map_fg, extras = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                    bc_rgb=bg_img, chunk=2048, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.head_model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=True,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    )

                # render torso
                cond_feat = self.model.cal_cond_feat(cond_wins, color=rgb_pred, euler=sample['euler'], trans=sample['trans'],with_att=True)
                rays_o, rays_d, _ = self.full_rays_sampler(H, W, focal, c2w_t0)
                rgb_pred_torso, disp_torso, acc_torso, last_weight_torso, rgb_map_fg_torso, extras_torso = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                    bc_rgb=bg_img,chunk=2048, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=False,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    euler=euler, euler_t0=euler_t0, trans=trans, trans_t0=trans_t0
                    )
                rgb_com = rgb_pred * last_weight_torso.unsqueeze(-1) + rgb_map_fg_torso

                model_out = {
                    "rgb_map" : rgb_com
                }
                return model_out
        else:
            # Training Phase
            if run_head_mode:
                # Run Head NeRF
                with torch.no_grad():
                    cond_feat = self.head_model.cal_cond_feat(cond_wins, with_att=True)

                    target = sample['head_img']
                    rays_o, rays_d, select_coords = self.rays_sampler(H, W, focal, c2w_t, n_rays=None, rect=sample['rect'], in_rect_percent=hparams['in_rect_percent'], iterations=self.global_step)
                    rgb_gt = self.rays_sampler.sample_pixels_from_img_with_select_coords(target, select_coords)
                    rgb_bc = self.rays_sampler.sample_pixels_from_img_with_select_coords(sample['bg_img'], select_coords)
                    rgb_pred, disp, acc, last_weight, rgb_map_fg, extras = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                        bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                        network_fn=self.head_model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                        run_head_mode=True,
                        c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                        )
                    losses_out['head_mse_loss'] = torch.mean((rgb_pred - rgb_gt) ** 2)
                    if 'rgb_map_coarse' in extras:
                        losses_out['head_mse_loss_coarse'] = torch.mean((extras['rgb_map_coarse'] - rgb_gt) ** 2)
                    model_out = {
                        "rgb_map": rgb_pred
                    }
            else:
                # Run Torso NeRF
                target = sample['gt_img']
                rect = [0, H/2, W, H/2] # only sample the lower part for torso

                rays_o, rays_d, select_coords = self.torso_rays_sampler(H, W, focal, c2w_t0, n_rays=None, rect=rect, 
                            in_rect_percent=hparams['in_rect_percent'])
                rays_o_head, rays_d_head, _ = self.rays_sampler(H, W, focal, c2w_t, select_coords=select_coords)
                rgb_gt = self.rays_sampler.sample_pixels_from_img_with_select_coords(target, select_coords)
                rgb_bc = self.rays_sampler.sample_pixels_from_img_with_select_coords(sample['bg_img'], select_coords)

                with torch.no_grad():
                    cond_feat = self.head_model.cal_cond_feat(cond_wins, with_att=True)
                    rgb_pred_head, disp_head, acc_head, last_weight_head, rgb_map_fg_head, extras_head = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o_head, rays_d=rays_d_head,
                    bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.head_model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=True,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    )

                cond_feat = self.model.cal_cond_feat(cond_wins, color=rgb_pred_head, euler=sample['euler'], trans=sample['trans'],with_att=True)

                rgb_pred_torso, disp_torso, acc_torso, last_weight_torso, rgb_map_fg_torso, extras_torso = render_dynamic_face(H, W, focal, cx, cy, rays_o=rays_o, rays_d=rays_d,
                    bc_rgb=rgb_bc,chunk=self.chunk, c2w=None, cond=cond_feat, near=near, far=far,
                    network_fn=self.model, N_samples=self.n_samples_per_ray, N_importance=self.n_samples_per_ray_fine,
                    run_head_mode=run_head_mode,
                    c2w_t=c2w_t, c2w_t0=c2w_t0,t=torch.tensor([0.,]).cuda(),
                    euler=euler, euler_t0=euler_t0, trans=trans, trans_t0=trans_t0
                    )

                rgb_com = rgb_pred_head * last_weight_torso.unsqueeze(-1) + rgb_map_fg_torso
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
    
       
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        #######################
        #      TorsoNeRF      #
        #######################
        loss_output, model_out = self.run_model(sample, infer=False, run_head_mode=False)
        def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)
        loss_output['torso_psnr'] = mse2psnr(loss_output['com_mse_loss'].detach())
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/model_coarse_torso': get_grad_norm(self.model.model_coarse),
            f'{prefix}/model_fine_torso': get_grad_norm(self.model.model_fine),
        }
        if hparams.get("use_color", False):
            grad_norm_dict[f'{prefix}/color_encoder_torso'] = get_grad_norm(self.model.color_encoder)
        return grad_norm_dict

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        sample['c2w_t0'] = convert_to_tensor(self.train_dataset.samples[0]['c2w'][:3]).float().to(sample['c2w'].device)
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False, run_head_mode=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = 1
        outputs = tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            idx_interval = (len(self.val_dataset)-1)//(hparams['num_valid_plots']-1)
            idx_lst = [i_plot*idx_interval for i_plot in range(hparams['num_valid_plots'])]
            sample = move_to_cuda(self.val_dataset[idx_lst[batch_idx]])
            sample['c2w_t0'] = convert_to_tensor(self.train_dataset.samples[0]['c2w'][:3]).float().to(sample['c2w'].device)
            infer_outputs = self.run_model(sample, infer=True)
            rgb_pred = infer_outputs['rgb_map']
            H, W = sample['H'], sample['W']
            img_pred = rgb_pred.reshape([H, W, 3])
            gen_dir = self.gen_dir
            base_fn = f"frame_{sample['idx']}"
            self.save_result(img_pred,  base_fn , gen_dir)
            target = sample['gt_img']
            img_gt = target.reshape([H, W, 3])
            if hparams['save_gt']:
                base_fn = f"frame_{sample['idx']}_gt"
                self.save_result(img_gt,  base_fn , gen_dir)
        return outputs