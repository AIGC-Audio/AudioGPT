from audio_to_face.utils.commons.base_task import BaseTask

import torch
import numpy as np
import os

from audio_to_face.utils.commons.base_task import BaseTask
from audio_to_face.utils.commons.dataset_utils import data_loader
from audio_to_face.utils.commons.hparams import hparams
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.commons.tensor_utils import tensors_to_scalars, convert_to_np
from audio_to_face.utils.nn.model_utils import print_arch, get_device_of_model, not_requires_grad
from audio_to_face.utils.nn.schedulers import ExponentialSchedule
from audio_to_face.utils.nn.grad import get_grad_norm

from audio_to_face.modules.audio2pose.models import Audio2PoseModel
from audio_to_face.modules.audio2pose.gmm_utils import GMMLogLoss
from audio_to_face.tasks.audio2pose.dataset_utils import Audio2PoseDataset

class Audio2PoseTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Audio2PoseDataset
        self.gmm_loss_fn = GMMLogLoss(ncenter=1, ndim=12, sigma_min=0.03)

    def build_model(self):
        self.model = Audio2PoseModel(hparams['reception_field'])
        return self.model

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        return optimizer

    def build_scheduler(self, optimizer):
        return ExponentialSchedule(optimizer, hparams['lr'], hparams['warmup_updates'])

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls()
        self.train_dl = train_dataset.get_dataloader()
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        val_dataset = self.dataset_cls()
        self.val_dl = val_dataset.get_dataloader()
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        val_dataset = self.dataset_cls(prefix='val')
        self.val_dl = val_dataset.get_dataloader()
        return self.val_dl

    ##########################
    # training and validation
    ##########################
    def run_model(self, sample):
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
        model_out = {}
        losses_out = {}
        audio_window = sample['audio_window']
        history_pose_and_velocity = sample['history_pose_and_velocity']
        target_pose_and_velocity = sample['target_pose_and_velocity']

        ret = self.model.forward(audio_window, history_pose_and_velocity)
        pred_pose_velocity_gmm_params = ret[:,-1, :]

        model_out['pred_pose_velocity_gmm_params'] = pred_pose_velocity_gmm_params
        losses_out['gmm_loss'] = self.gmm_loss_fn(pred_pose_velocity_gmm_params.unsqueeze(1), target_pose_and_velocity.unsqueeze(1))
        losses_out['history_gmm_loss'] = self.gmm_loss_fn(ret[:-1], history_pose_and_velocity[1:])

        return losses_out, model_out

            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'gmm_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
        
    def get_grad(self, opt_idx):
        grad_dict = {
            'grad/model': get_grad_norm(self.model),
        }
        return grad_dict
