import os

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from tasks.base_task import BaseTask
from tasks.base_task import data_loader
from tasks.vocoder.dataset_utils import VocoderDataset, EndlessDistributedSampler
from utils.hparams import hparams


class VocoderBaseTask(BaseTask):
    def __init__(self):
        super(VocoderBaseTask, self).__init__()
        self.max_sentences = hparams['max_sentences']
        self.max_valid_sentences = hparams['max_valid_sentences']
        if self.max_valid_sentences == -1:
            hparams['max_valid_sentences'] = self.max_valid_sentences = self.max_sentences
        self.dataset_cls = VocoderDataset

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls('train', shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_sentences, hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls('valid', shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_valid_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls('test', shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_valid_sentences)

    def build_dataloader(self, dataset, shuffle, max_sentences, endless=False):
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        sampler_cls = DistributedSampler if not endless else EndlessDistributedSampler
        train_sampler = sampler_cls(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            collate_fn=dataset.collater,
            batch_size=max_sentences,
            num_workers=dataset.num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )

    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'],
                                    f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)

    def test_end(self, outputs):
        return {}
