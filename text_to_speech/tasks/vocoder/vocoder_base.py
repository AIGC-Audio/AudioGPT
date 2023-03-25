import os
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler
from tasks.vocoder.dataset_utils import VocoderDataset, EndlessDistributedSampler
from text_to_speech.utils.audio.io import save_wav
from text_to_speech.utils.commons.base_task import BaseTask
from text_to_speech.utils.commons.dataset_utils import data_loader
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.commons.tensor_utils import tensors_to_scalars


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
        valid_dataset = self.dataset_cls('test', shuffle=False)
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

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(self.model_gen.parameters(), lr=hparams['lr'],
                                          betas=[hparams['adam_b1'], hparams['adam_b2']])
        optimizer_disc = torch.optim.AdamW(self.model_disc.parameters(), lr=hparams['lr'],
                                           betas=[hparams['adam_b1'], hparams['adam_b2']])
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[0],
                **hparams["generator_scheduler_params"]),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[1],
                **hparams["discriminator_scheduler_params"]),
        }

    def validation_step(self, sample, batch_idx):
        outputs = {}
        total_loss, loss_output = self._training_step(sample, batch_idx, 0)
        outputs['losses'] = tensors_to_scalars(loss_output)
        outputs['total_loss'] = tensors_to_scalars(total_loss)

        if self.global_step % hparams['valid_infer_interval'] == 0 and \
                batch_idx < 10:
            mels = sample['mels']
            y = sample['wavs']
            f0 = sample['f0']
            y_ = self.model_gen(mels, f0)
            for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
                wav_pred = wav_pred / wav_pred.abs().max()
                if self.global_step == 0:
                    wav_gt = wav_gt / wav_gt.abs().max()
                    self.logger.add_audio(f'wav_{batch_idx}_{idx}_gt', wav_gt, self.global_step,
                                          hparams['audio_sample_rate'])
                self.logger.add_audio(f'wav_{batch_idx}_{idx}_pred', wav_pred, self.global_step,
                                      hparams['audio_sample_rate'])
        return outputs

    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'],
                                    f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)

    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        f0 = sample['f0']
        loss_output = {}
        y_ = self.model_gen(mels, f0)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
            wav_gt = wav_gt.clamp(-1, 1)
            wav_pred = wav_pred.clamp(-1, 1)
            save_wav(
                wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_gt.wav',
                hparams['audio_sample_rate'])
            save_wav(
                wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_pred.wav',
                hparams['audio_sample_rate'])
        return loss_output

    def test_end(self, outputs):
        return {}

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step // hparams['accumulate_grad_batches'])
        else:
            self.scheduler['disc'].step(self.global_step // hparams['accumulate_grad_batches'])
