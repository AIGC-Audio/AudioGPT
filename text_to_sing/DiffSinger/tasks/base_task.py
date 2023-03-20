import glob
import re
import subprocess
from datetime import datetime

import matplotlib

matplotlib.use('Agg')

from utils.hparams import hparams, set_hparams
import random
import sys
import numpy as np
import torch.distributed as dist
from pytorch_lightning.loggers import TensorBoardLogger
from utils.pl_utils import LatestModelCheckpoint, BaseTrainer, data_loader, DDP
from torch import nn
import torch.utils.data
import utils
import logging
import os

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], hparams['max_frames'])
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
                # 先random, 然后稳定排序, 保证排序后同长度的数据顺序是依照random permutation的 (被其随机打乱).
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))


class BaseTask(nn.Module):
    def __init__(self, *args, **kwargs):
        # dataset configs
        super(BaseTask, self).__init__(*args, **kwargs)
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.logger = None
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.example_input_array = None

        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences

        self.model = None
        self.training_losses_meter = None

    ###########
    # Training, validation and testing
    ###########
    def build_model(self):
        raise NotImplementedError

    def load_ckpt(self, ckpt_base_dir, current_model_name=None, model_name='model', force=True, strict=True):
        # This function is updated on 2021.12.13
        if current_model_name is None:
            current_model_name = model_name
        utils.load_ckpt(self.__getattr__(current_model_name), ckpt_base_dir, current_model_name, force, strict)

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        try:
            log_outputs['lr'] = self.scheduler.get_lr()
            if isinstance(log_outputs['lr'], list):
                log_outputs['lr'] = log_outputs['lr'][0]
        except:
            pass

        # log_outputs['all_loss'] = total_loss.item()
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        print(f"\n==============\n "
              f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}"
              f"\n==============\n")

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: dict
        """
        raise NotImplementedError

    def _validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"\n==============\n "
              f"valid results: {loss_output}"
              f"\n==============\n")
        return {
            'log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        return [optm]

    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ###########
    # Running configuration
    ###########

    @classmethod
    def start(cls):
        set_hparams()
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        task = cls()
        work_dir = hparams['work_dir']
        trainer = BaseTrainer(checkpoint_callback=LatestModelCheckpoint(
                                  filepath=work_dir,
                                  verbose=True,
                                  monitor='val_loss',
                                  mode='min',
                                  num_ckpt_keep=hparams['num_ckpt_keep'],
                                  save_best=hparams['save_best'],
                                  period=1 if hparams['save_ckpt'] else 100000
                              ),
                              logger=TensorBoardLogger(
                                  save_dir=work_dir,
                                  name='lightning_logs',
                                  version='lastest'
                              ),
                              gradient_clip_val=hparams['clip_grad_norm'],
                              val_check_interval=hparams['val_check_interval'],
                              row_log_interval=hparams['log_interval'],
                              max_updates=hparams['max_updates'],
                              num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams[
                                  'validate'] else 10000,
                              accumulate_grad_batches=hparams['accumulate_grad_batches'])
        if not hparams['infer']:  # train
            t = datetime.now().strftime('%Y%m%d%H%M%S')
            code_dir = f'{work_dir}/codes/{t}'
            subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
            for c in hparams['save_codes']:
                subprocess.check_call(f'cp -r "{c}" "{code_dir}/"', shell=True)
            print(f"| Copied codes to {code_dir}.")
            trainer.checkpoint_callback.task = task
            trainer.fit(task)
        else:
            trainer.test(task)

    def configure_ddp(self, model, device_ids):
        model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        if dist.get_rank() != 0 and not hparams['debug']:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        return model

    def training_end(self, *args, **kwargs):
        return None

    def init_ddp_connection(self, proc_rank, world_size):
        set_hparams(print_hparams=False)
        # guarantees unique ports across jobs from same grid search
        default_port = 12910
        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        root_node = '127.0.0.2'
        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    @data_loader
    def train_dataloader(self):
        return None

    @data_loader
    def test_dataloader(self):
        return None

    @data_loader
    def val_dataloader(self):
        return None

    def on_load_checkpoint(self, checkpoint):
        pass

    def on_save_checkpoint(self, checkpoint):
        pass

    def on_sanity_check_start(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_start(self, batch):
        pass

    def on_batch_end(self):
        pass

    def on_pre_performance_check(self):
        pass

    def on_post_performance_check(self):
        pass

    def on_before_zero_grad(self, optimizer):
        pass

    def on_after_backward(self):
        pass

    def backward(self, loss, optimizer):
        loss.backward()

    def grad_norm(self, norm_type):
        results = {}
        total_norm = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    results['grad_{}_norm_{}'.format(norm_type, name)] = grad
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['grad_{}_norm_total'.format(norm_type)] = grad
        return results
