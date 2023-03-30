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
from audio_to_face.utils.nn.grad import get_grad_norm
from audio_to_face.utils.nn.model_utils import num_params
from audio_to_face.utils.nn.schedulers import ExponentialSchedule, ExponentialScheduleWithAudattNet

from audio_to_face.tasks.nerfs.adnerf import ADNeRFTask
from audio_to_face.modules.nerfs.lm3d_nerf.lm3d_nerf import Lm3dNeRF
from audio_to_face.modules.nerfs.commons.volume_rendering import render_dynamic_face


class Lm3dNeRFTask(ADNeRFTask):
    def __init__(self):
        super().__init__()

    def build_model(self):
        self.model = Lm3dNeRF(hparams)
        if hparams['with_att']:
            self.lmatt_encoder_params = [p for p in self.model.lmatt_encoder.parameters() if p.requires_grad]
            self.gen_params_except_lmatt_encoder = [p for k, p in self.model.named_parameters() if (('lmatt_encoder' not in k) and p.requires_grad)]        
        else:
            self.gen_params = [p for k, p in self.model.named_parameters() if p.requires_grad]        
        return self.model
    
    def build_optimizer(self, model):
        if hparams['with_att']:
            self.optimizer = torch.optim.Adam(
                self.gen_params_except_lmatt_encoder,
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
            self.optimizer.add_param_group({
                'params': self.lmatt_encoder_params,
                'lr': hparams['lr'] * 5,
                'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2'])
            })
        else:
            self.optimizer = torch.optim.Adam(
                self.gen_params,
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        return self.optimizer

    def build_scheduler(self, optimizer):
        if hparams['with_att']:
            return ExponentialScheduleWithAudattNet(optimizer, hparams['lr'], hparams['warmup_updates'])
        else:
            return ExponentialSchedule(optimizer, hparams['lr'], hparams['warmup_updates'])
    
    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/model_coarse': get_grad_norm(self.model.model_coarse),
            f'{prefix}/model_fine': get_grad_norm(self.model.model_fine),
            f'{prefix}/lm_encoder': get_grad_norm(self.model.lm_encoder),
        }
        if hparams['with_att']:
            grad_norm_dict[f'{prefix}/lmatt_encoder'] = get_grad_norm(self.model.lmatt_encoder)
        return grad_norm_dict
