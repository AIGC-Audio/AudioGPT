import torch
import torch.nn as nn
import numpy as np
import os
import cv2

from audio_to_face.modules.nerfs.commons.volume_rendering import render_dynamic_face
from audio_to_face.modules.nerfs.adnerf.adnerf import ADNeRF
from audio_to_face.modules.nerfs.commons.ray_samplers import UniformRaySampler, FullRaySampler, PatchRaySampler

from audio_to_face.utils.commons.image_utils import to8b
from audio_to_face.utils.commons.base_task import BaseTask
from audio_to_face.utils.commons.dataset_utils import data_loader
from audio_to_face.utils.commons.hparams import hparams
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from audio_to_face.utils.nn.model_utils import print_arch, num_params
from audio_to_face.utils.nn.schedulers import ExponentialScheduleWithAudattNet
from audio_to_face.utils.nn.grad import get_grad_norm

from audio_to_face.tasks.nerfs.dataset_utils import NeRFDataset


class ADNeRFTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.chunk = 1024
        self.no_smo_iterations = hparams['no_smo_iterations']
        self.n_samples_per_ray = hparams['n_samples_per_ray']
        self.n_samples_per_ray_fine = hparams['n_samples_per_ray_fine']
        self.n_rays = hparams['n_rays']
        self.rays_sampler = UniformRaySampler(N_rays=self.n_rays)
        self.full_rays_sampler = FullRaySampler()
        self.dataset_cls = NeRFDataset
        self.train_dataset = self.dataset_cls(prefix='train')
        self.val_dataset = self.dataset_cls(prefix='val')

    def build_model(self):
        self.model = ADNeRF(hparams)
        self.audatt_net_params = [p for p in self.model.audatt_net.parameters() if p.requires_grad]
        self.gen_params_except_audatt_net = [p for k, p in self.model.named_parameters() if (('audatt_net' not in k) and p.requires_grad)]        
        return self.model

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
            
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

    @data_loader
    def train_dataloader(self):
        self.train_dl = torch.utils.data.DataLoader(self.train_dataset,collate_fn=self.train_dataset.collater,
                                            batch_size=1, shuffle=True, 
                                            num_workers=0, pin_memory=True)
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        self.val_dl = torch.utils.data.DataLoader(self.val_dataset,collate_fn=self.val_dataset.collater,
                                            batch_size=1, shuffle=True, 
                                            num_workers=0, pin_memory=True)
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        self.val_dl = torch.utils.data.DataLoader(self.val_dataset,collate_fn=self.val_dataset.collater,
                                            batch_size=1, shuffle=False, 
                                            num_workers=0, pin_memory=True)
        return self.val_dl
        
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
        
        with_att = self.global_step >= self.no_smo_iterations
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
    
    ##########################
    # training 
    ##########################
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)
        loss_output['head_psnr'] = mse2psnr(loss_output['mse_loss'].detach())
        return total_loss, loss_output
    
    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/model_coarse': get_grad_norm(self.model.model_coarse),
            f'{prefix}/model_fine': get_grad_norm(self.model.model_fine),
            f'{prefix}/aud_net': get_grad_norm(self.model.aud_net),
            f'{prefix}/audatt_net': get_grad_norm(self.model.audatt_net),
        }
        return grad_norm_dict
        
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    #####################
    # Validation
    #####################
    def validation_start(self):
        if self.global_step % hparams['valid_infer_interval'] == 0:
            self.gen_dir = os.path.join(hparams['work_dir'], f'validation_results/validation_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            os.makedirs(self.gen_dir, exist_ok=True)
            os.makedirs(f'{self.gen_dir}/imgs', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = 1
        outputs = tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            # idx_lst = [291,156,540,113,28]
            num_val_samples = len(self.val_dataset)
            interval = (num_val_samples-1) // 4
            idx_lst = [i * interval for i in range(5)]
            sample = move_to_cuda(self.val_dataset[idx_lst[batch_idx]])
            infer_outputs = self.run_model(sample, infer=True)
            rgb_pred = infer_outputs['rgb_map']
            H, W = sample['H'], sample['W']
            img_pred = rgb_pred.reshape([H, W, 3])
            gen_dir = self.gen_dir
            base_fn = f"frame_{sample['idx']}"
            self.save_result(img_pred,  base_fn , gen_dir)
            target = sample['head_img']
            img_gt = target.reshape([H, W, 3])
            if hparams['save_gt']:
                base_fn = f"frame_{sample['idx']}_gt"
                self.save_result(img_gt,  base_fn , gen_dir)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)

    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/imgs', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        outputs = self.run_model(sample, infer=True)
        rgb_pred = outputs['rgb_map']
        H, W = sample['H'], sample['W']
        img_pred = rgb_pred.reshape([H, W, 3])
        gen_dir = self.gen_dir
        base_fn = f"frame_{sample['idx']}"
        self.save_result(img_pred,  base_fn , gen_dir)
        target = sample['gt_img'] if hparams['use_pos_deform'] else sample['head_img']
        img_gt = target.reshape([H, W, 3])
        if hparams['save_gt']:
            base_fn = f"frame_{sample['idx']}_gt"
            self.save_result(img_gt,  base_fn , gen_dir)
        outputs['losses'] = (img_gt - img_pred).mean()
        return outputs

    def test_end(self, outputs):
        pass

    #####################
    # Visualization utils
    #####################
    @staticmethod
    def save_result(rgb, base_fname, gen_dir):
        rgb = convert_to_np(rgb * 255.).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{gen_dir}/imgs/{base_fname}.jpg", bgr)
