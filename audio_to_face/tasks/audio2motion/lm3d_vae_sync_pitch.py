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

from audio_to_face.modules.audio2motion.vae import PitchContourVAEModel
from audio_to_face.tasks.audio2motion.dataset_utils.lrs3_dataset import LRS3SeqDataset
from audio_to_face.tasks.syncnet.lm3d_syncnet import SyncNetTask

from audio_to_face.data_util.face3d_helper import Face3DHelper

class VAESyncAudio2MotionTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = LRS3SeqDataset
        self.enable_sync = False # enables when sync loss is lower than 0.5!
        self.face3d_helper = Face3DHelper(use_gpu=True)

    def build_model(self):
        self.syncnet_task = SyncNetTask()
        self.syncnet_task.build_model()
        load_ckpt(self.syncnet_task.model, hparams["syncnet_work_dir"], steps=hparams["syncnet_ckpt_steps"])
        not_requires_grad(self.syncnet_task)
        self.syncnet_task.eval()

        self.model = PitchContourVAEModel(in_out_dim=68*3)
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
        train_dataset = self.dataset_cls(prefix='train')
        self.train_dl = train_dataset.get_dataloader()
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        val_dataset = self.dataset_cls(prefix='val')
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
    def run_model(self, sample, infer=False, temperature=1.0, sync_batch_size=1024):
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
        if infer:
            self.model(sample, model_out, train=False, temperature=temperature)
        else:
            sample['y'] = sample['idexp_lm3d']
            self.model(sample, model_out, train=True)

        if not infer:
            # forward the syncnet to get sync_loss
            losses_out = {}
            pred_lm3d = model_out['pred']
            pred_lm3d = pred_lm3d.reshape(pred_lm3d.size(0), pred_lm3d.size(1), 68, 3)
        
            _, pred_mouth_lm3d = self.face3d_helper.get_eye_mouth_lm_from_lm3d_batch(pred_lm3d)
            syncnet_sample = {
                'mouth_idexp_lm3d': pred_mouth_lm3d.reshape(pred_mouth_lm3d.size(0), pred_mouth_lm3d.size(1), -1),
                'hubert': sample['hubert'],
                'y_mask': model_out['mask'],
            }
            syncnet_out = self.syncnet_task.run_model(syncnet_sample, infer=True, batch_size=sync_batch_size)
            losses_out['sync'] = syncnet_out['sync_loss']

            x_gt = sample['idexp_lm3d']
            x_pred = model_out['pred']
            x_mask = model_out['mask']
            losses_out['mse'] = self.mse_loss(x_gt, x_pred, x_mask)
            losses_out['continuity'] = self.continuity_loss(x_gt, x_pred, x_mask)
            losses_out['kl'] = model_out['loss_kl']
            return losses_out, model_out
        else:
            return model_out
            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'kl': hparams['lambda_kl'],
            'mse': 1.0,
            'continuity': 3.0,
            'sync': hparams.get('lambda_sync', 0.01) if self.enable_sync else 0.
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False, sync_batch_size=10000)
        outputs = tensors_to_scalars(outputs)
        if outputs['losses']['sync'] <= 0.75 and not self.enable_sync:
            self.enable_sync = True
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
        
    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        outputs = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        pred_exp = model_out['pred']
        self.save_result(pred_exp,  "pred_exp_val" , self.gen_dir)
        if hparams['save_gt']:
            base_fn = f"gt_exp_val"
            self.save_result(sample['exp'],  base_fn , self.gen_dir)
        return outputs

    def test_end(self, outputs):
        pass

    @staticmethod
    def save_result(exp_arr, base_fname, gen_dir):
        exp_arr = convert_to_np(exp_arr)
        np.save(f"{gen_dir}/{base_fname}.npy", exp_arr)
    
    def get_grad(self, opt_idx):
        grad_dict = {
            'grad/model': get_grad_norm(self.model),
        }
        return grad_dict
    
    def mse_loss(self, x_gt, x_pred, x_mask):
        # mean squared error, l2 loss
        error = (x_pred - x_gt) * x_mask[:,:, None]
        num_frame = x_mask.sum()
        n_dim = 68*3
        return (error ** 2).sum() / (num_frame * n_dim)
    
    def mae_loss(self, x_gt, x_pred, x_mask):
        # mean absolute error, l1 loss
        error = (x_pred - x_gt) * x_mask[:,:, None]
        num_frame = x_mask.sum()
        n_dim = 68*3
        return error.abs().sum() / (num_frame * n_dim)

    def continuity_loss(self, x_gt, x_pred, x_mask):
        # continuity loss, borrowed from <FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning>
        diff_x_pred = x_pred[:,1:] - x_pred[:,:-1]
        diff_x_gt = x_gt[:,1:] - x_gt[:,:-1]
        error = (diff_x_pred[:,:,:] - diff_x_gt[:,:,:]) * x_mask[:,1:,None]
        init_error = x_pred[:,0,:] - x_gt[:,0,:]
        num_frame = x_mask.sum()
        n_dim = 68*3
        return (error.pow(2).sum() + init_error.pow(2).sum()) / (num_frame * n_dim)