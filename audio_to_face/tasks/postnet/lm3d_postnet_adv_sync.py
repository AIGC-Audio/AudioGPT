import torch
import importlib

from audio_to_face.utils.commons.base_task import BaseTask
from audio_to_face.utils.commons.dataset_utils import data_loader
from audio_to_face.utils.commons.hparams import hparams
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.commons.tensor_utils import tensors_to_scalars, convert_to_np
from audio_to_face.utils.nn.model_utils import print_arch

from audio_to_face.modules.postnet.models import CNNPostNet, MLPDiscriminator
from audio_to_face.tasks.audio2motion.lm3d_vae_sync import VAESyncAudio2MotionTask
from audio_to_face.tasks.postnet.dataset_utils import PostnetDataset
from audio_to_face.tasks.syncnet.lm3d_syncnet import SyncNetTask

from audio_to_face.data_util.face3d_helper import Face3DHelper


class PostnetAdvSyncTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.audio2motion_task = self.build_audio2motion_task()
        self.syncnet_task = self.build_syncnet_task()
        self.build_disc_model()
        self.dataset_cls = PostnetDataset
        self.face3d_helper = Face3DHelper(use_gpu=True)

    def build_audio2motion_task(self):
        assert hparams['audio2motion_task_cls'] != ''
        pkg = ".".join(hparams["audio2motion_task_cls"].split(".")[:-1])
        cls_name = hparams["audio2motion_task_cls"].split(".")[-1]
        task_cls = getattr(importlib.import_module(pkg), cls_name)
        self.audio2motion_task = task_cls()
        self.audio2motion_task.build_model()
        audio2motion_work_dir = hparams['audio2motion_work_dir']
        audio2motion_ckpt_steps = hparams["audio2motion_ckpt_steps"]
        load_ckpt(self.audio2motion_task.model, audio2motion_work_dir, 'model', steps=audio2motion_ckpt_steps)
        self.audio2motion_task.eval()
        return self.audio2motion_task

    def build_syncnet_task(self):
        self.syncnet_task = SyncNetTask()
        self.syncnet_task.build_model()
        syncnet_work_dir = hparams["syncnet_work_dir"]
        syncnet_ckpt_steps = hparams["syncnet_ckpt_steps"]
        load_ckpt(self.syncnet_task.model, syncnet_work_dir, 'model', steps=syncnet_ckpt_steps)
        for p in self.syncnet_task.parameters():
            p.requires_grad = False
        self.syncnet_task.eval()
        return self.syncnet_task

    def build_model(self):
        self.model = CNNPostNet(in_out_dim=68*3)
        print_arch(self.model)
        return self.model

    def build_disc_model(self):
        self.disc_model = MLPDiscriminator(in_dim=68*3)

    def build_optimizer(self, model):
        self.optimizer_gen = torch.optim.RMSprop(self.model.parameters(),
                            lr=hparams['postnet_lr'],)
        self.optimizer_disc = torch.optim.RMSprop(self.disc_model.parameters(),
                            lr=hparams['postnet_disc_lr'],)

        return [self.optimizer_gen, self.optimizer_disc]


    def build_scheduler(self, optimizer):
        return [
            VAESyncAudio2MotionTask.build_scheduler(self, optimizer[0]),
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1],
                 **hparams["discriminator_scheduler_params"]),
        ]

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])

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
    def run_model(self, sample, infer=False, temperature=1.0):
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
        losses_out = {}
        lrs3_batch = {
            'x_mask': sample['x_mask'],
            'y_mask': sample['y_mask'],
            'hubert': sample['hubert'],
        }
        ### Perform the audio2motion first
        with torch.no_grad():
            model_out = self.audio2motion_task.run_model(lrs3_batch, infer=True, temperature=temperature)
            raw_pred_lm3d = model_out['pred']
        
        ### Then forward the PostNet
        if infer:
            refine_pred_lm3d = self.model(raw_pred_lm3d)
            model_out = {
                'refine_lm3d': refine_pred_lm3d,
                'raw_lm3d': raw_pred_lm3d
            }
            return model_out
        else:
            person_batch = sample['person_ds']
            gt_pred_lm3d_for_person_ds = person_batch['idexp_lm3d'] 
            with torch.no_grad():
                model_out = self.audio2motion_task.run_model(person_batch, infer=True, temperature=temperature)
                raw_pred_lm3d_for_person_ds = model_out['pred']
            refine_pred_lm3d_for_person_ds = self.model(raw_pred_lm3d_for_person_ds) * person_batch['y_mask'].unsqueeze(-1)
            if hparams.get("loss_type", 'mse') == 'mse':
                losses_out['mse'] = (gt_pred_lm3d_for_person_ds - refine_pred_lm3d_for_person_ds).pow(2).sum() / (person_batch['y_mask'].sum() * 68*3)
            else:
                losses_out['mae'] = (gt_pred_lm3d_for_person_ds - refine_pred_lm3d_for_person_ds).abs().sum() / (person_batch['y_mask'].sum() * 68*3)

            refine_pred_lm3d = self.model(raw_pred_lm3d)

            ### Calculate Syncnet Loss
            refine_pred_lm3d_ = refine_pred_lm3d.reshape(refine_pred_lm3d.size(0), refine_pred_lm3d.size(1), 68, 3)
            _, refine_mouth_lm3d = self.face3d_helper.get_eye_mouth_lm_from_lm3d_batch(refine_pred_lm3d_)
            syncnet_sample = {
                'idexp_lm3d': refine_pred_lm3d.reshape(refine_pred_lm3d.size(0), refine_pred_lm3d.size(1), -1),
                'mouth_idexp_lm3d': refine_mouth_lm3d.reshape(refine_mouth_lm3d.size(0), refine_mouth_lm3d.size(1), -1),
                'hubert': sample['hubert'],
                'y_mask': sample['y_mask']
            }
            syncnet_out = self.syncnet_task.run_model(syncnet_sample, infer=True, batch_size=1024)
            sync_loss = syncnet_out['sync_loss']
            losses_out['sync'] = sync_loss

            model_out = {
                'refine_lm3d': refine_pred_lm3d
            }
            return losses_out, model_out
      
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["postnet_disc_start_steps"] and hparams['postnet_lambda_adv'] > 0
        if optimizer_idx == 0:
            loss_output, model_out = self.run_model(sample)
            loss_weights = {
                'mse': hparams['postnet_lambda_mse'],
            }

            pred = model_out['refine_lm3d']
            self.pred = pred.detach()
            if disc_start:
                disc_conf_neg = self.disc_model(x=pred)[0]
                loss_output['adv'] = (1 - disc_conf_neg).pow(2).mean()
                loss_weights['adv'] = hparams['postnet_lambda_adv']
                loss_weights['sync'] = hparams['postnet_lambda_sync']
        else:
            # train the discriminator
            if self.global_step % hparams['postnet_disc_interval'] == 0:
                pred = self.pred
                p_ = self.disc_model(x=pred)[0]
                person_idexp_normalized = sample['person_ds']['idexp_lm3d'] 
                p = self.disc_model(x=person_idexp_normalized)[0]
                loss_output['disc_neg_conf'] = p_.detach().mean().item()
                loss_output['disc_pos_conf'] = p.detach().mean().item()

                loss_output["disc_fake_loss"] = (p_ - p_.new_zeros(p_.size())).pow(2).mean()
                loss_output["disc_true_loss"] = (p - p.new_ones(p.size())).pow(2).mean()
            else:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        return total_loss, loss_output

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs = tensors_to_scalars(outputs)
        return outputs

