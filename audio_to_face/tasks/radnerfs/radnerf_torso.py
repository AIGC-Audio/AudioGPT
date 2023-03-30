import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import lpips
import matplotlib.pyplot as plt

from audio_to_face.modules.radnerfs.radnerf import RADNeRF
from audio_to_face.modules.radnerfs.radnerf_torso import RADNeRFTorso
from audio_to_face.tasks.radnerfs.radnerf import RADNeRFTask

from audio_to_face.utils.commons.image_utils import to8b
from audio_to_face.utils.commons.base_task import BaseTask
from audio_to_face.utils.commons.dataset_utils import data_loader
from audio_to_face.utils.commons.hparams import hparams, set_hparams
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from audio_to_face.utils.nn.model_utils import print_arch, num_params, not_requires_grad
from audio_to_face.utils.nn.schedulers import ExponentialScheduleForRADNeRFTorso
from audio_to_face.utils.nn.grad import get_grad_norm

from audio_to_face.tasks.radnerfs.dataset_utils import RADNeRFDataset


class RADNeRFTorsoTask(RADNeRFTask):
    def __init__(self):
        super().__init__()

    def build_model(self):
        hparams = set_hparams('audio_to_face/checkpoints/May/lm3d_radnerf_torso/config.yaml')

        self.model = RADNeRFTorso(hparams) 
        # todo: load state_dict in RADNeRF
        head_model = RADNeRF(hparams)
        load_ckpt(head_model, hparams['head_model_dir'])
        print(f"Loaded Head Model from {hparams['head_model_dir']}")
        self.model.load_state_dict(head_model.state_dict(), strict=False)
        print(f"Loaded state_dict of Head Model to the RADNeRFTorso Model")
        del head_model

        self.torso_embedders_params = [p for k, p in self.model.named_parameters() if p.requires_grad and 'torso_embedder' in k]
        self.torso_network_params = [p for k, p in self.model.named_parameters() if (p.requires_grad and 'torso_embedder' not in k and 'torso' in k)]
        for k, p in self.model.named_parameters():
            if 'torso' not in k:
                not_requires_grad(p)

        self.model.poses = self.train_dataset.poses
        return self.model

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
            
    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.torso_network_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            eps=1e-15)
        self.optimizer.add_param_group({
            'params': self.torso_embedders_params,
            'lr': hparams['lr'] * 10,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })
        return self.optimizer
    
    def build_scheduler(self, optimizer):
        return ExponentialScheduleForRADNeRFTorso(optimizer, hparams['lr'], hparams['warmup_updates'])

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
        cond_wins = sample['cond_wins']
        rays_o = sample['rays_o'] # [B, N, 3]
        rays_d = sample['rays_d'] # [B, N, 3]
        bg_coords = sample['bg_coords'] # [1, N, 2]
        poses = sample['pose'] # [B, 6]
        idx = sample['idx'] # [B]
        bg_color = sample['bg_img']
        H, W = sample['H'], sample['W']

        cond_inp = cond_wins

        if not infer:
            model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **hparams)
            if hparams['torso_train_mode'] == 1:
                pred_rgb = model_out['torso_rgb_map'] 
                gt_rgb = sample['bg_torso_img'] # the target is bg_torso_img
            else:
                pred_rgb = model_out['rgb_map'] # todo: try whole image 
                gt_rgb = sample['gt_img'] # todo: try gt_image

            losses_out = {}

            losses_out['torso_mse_loss'] = torch.mean((pred_rgb - gt_rgb) ** 2) # [B, N, 3] -->  scalar

            alphas = model_out['torso_alpha_map'].clamp(1e-5, 1 - 1e-5)
            losses_out['torso_weights_entropy_loss'] = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
        
            return losses_out, model_out
            
        else:
            # infer phase, generate the whole image
            model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=False, force_all_rays=True, **hparams)
            # calculate val loss
            if 'gt_img' in sample:
                gt_rgb = sample['gt_img']
                pred_rgb = model_out['rgb_map']
                model_out['mse_loss'] = torch.mean((pred_rgb - gt_rgb) ** 2) # [B, N, 3] -->  scalar
            return model_out

    ##########################
    # training 
    ##########################
    def _training_step(self, sample, batch_idx, optimizer_idx):
        outputs = {}
        self.train_dataset.global_step = self.global_step
        if self.global_step % hparams['update_extra_interval'] == 0:
            self.model.update_extra_state()

        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'torso_mse_loss': 1.0,
            'torso_weights_entropy_loss': hparams['lambda_weights_entropy'],
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)
        loss_output['image_psnr'] = mse2psnr(loss_output['torso_mse_loss'].detach())
        outputs.update(loss_output)

        if (self.global_step+1) % hparams['tb_log_interval'] == 0:
            density_grid_info = {
                "density_grid_info/min_density_torso": self.model.density_grid_torso.min().item(),
                "density_grid_info/max_density_torso": self.model.density_grid_torso.max().item(),
                "density_grid_info/mean_density_torso": self.model.mean_density_torso,
                "density_grid_info/occupancy_rate_torso": (self.model.density_grid_torso > min(self.model.mean_density_torso, self.model.density_thresh_torso)).sum() / (128**3 * self.model.cascade), 
                "density_grid_info/step_mean_count_torso": self.model.mean_count
            }
            outputs.update(density_grid_info)
        return total_loss, outputs
    
    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/torso_embedders_params': get_grad_norm(self.torso_embedders_params),
            f'{prefix}/torso_network_params': get_grad_norm(self.torso_network_params ),
        }
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
        return grad_norm_dict
        
