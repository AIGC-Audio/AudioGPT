import torch
import random 

from audio_to_face.utils.commons.base_task import BaseTask
from audio_to_face.utils.commons.dataset_utils import data_loader
from audio_to_face.utils.commons.hparams import hparams
from audio_to_face.utils.commons.tensor_utils import tensors_to_scalars
from audio_to_face.utils.nn.model_utils import print_arch
from audio_to_face.utils.nn.schedulers import CosineSchedule

from audio_to_face.modules.syncnet.models import LandmarkHubertSyncNet
from audio_to_face.tasks.audio2motion.dataset_utils.lrs3_dataset import LRS3SeqDataset

class SyncNetTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = LRS3SeqDataset

    def build_model(self):
        lm_dim = 20*3
        self.model = LandmarkHubertSyncNet(lm_dim)
        print_arch(self.model)
        return self.model

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        return optimizer

    def build_scheduler(self, optimizer):
        return CosineSchedule(optimizer, hparams['lr'], warmup_updates=0, total_updates=hparams['max_updates'])

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
    def run_model(self, sample, infer=False, batch_size=1024):
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
        mouth_lm3d = sample['mouth_idexp_lm3d']
        mel = sample['hubert']
        
        y_mask = sample['y_mask']
        y_len = y_mask.sum(dim=1)
        mouth_lst, mel_lst, label_lst = [], [], []
        while len(mouth_lst) < batch_size:
            for i in range(mouth_lm3d.shape[0]):
                if not infer:
                    is_pos_sample = random.choice([True, False])
                else:
                    is_pos_sample = True
                exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                mouth_clip = mouth_lm3d[i, exp_idx: exp_idx+5]
                assert mouth_clip.shape[0]==5, f"exp_idx={exp_idx},y_len={y_len[i]}"
                if is_pos_sample:
                    mel_clip = mel[i, exp_idx*2: exp_idx*2 + 10]
                    label_lst.append(1.)
                else:
                    if random.random() < 0.25:
                        wrong_spk_idx = random.randint(a=0, b=len(y_len)-1)
                        wrong_exp_idx = random.randint(a=0, b=y_len[wrong_spk_idx]-1-5)
                        while wrong_exp_idx == exp_idx:
                            wrong_exp_idx = random.randint(a=0, b=y_len[wrong_spk_idx]-1-5)
                        mel_clip = mel[wrong_spk_idx, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10
                    elif random.random() < 0.5:
                        wrong_exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                        while wrong_exp_idx == exp_idx:
                            wrong_exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                        mel_clip = mel[i, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10
                    else:
                        left_offset = max(-5, -exp_idx)
                        right_offset = min(5, (y_len[i]-5-exp_idx))
                        exp_offset = random.randint(a=left_offset, b=right_offset)
                        while abs(exp_offset) <= 1:
                            exp_offset = random.randint(a=left_offset, b=right_offset)
                        wrong_exp_idx = exp_offset + exp_idx
                        mel_clip = mel[i, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10, y_len[i]-wrong_exp_idx
                    mel_clip = mel[i, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                    label_lst.append(0.)
                mouth_lst.append(mouth_clip)
                mel_lst.append(mel_clip)
        mel_clips = torch.stack(mel_lst)
        mouth_clips = torch.stack(mouth_lst)
        labels = torch.tensor(label_lst).float().to(mel_clips.device)

        audio_embedding, mouth_embedding = self.model(mel_clips, mouth_clips)
        sync_loss, cosine_sim = self.model.cal_sync_loss(audio_embedding, mouth_embedding, labels)
        if not infer:
            losses_out = {}
            model_out = {}
            losses_out['sync_loss'] = sync_loss
            model_out['cosine_sim'] = cosine_sim
            return losses_out, model_out
        else:
            model_out['sync_loss'] = sync_loss
            return model_out
            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample, infer=False)
        loss_weights = {}
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False, batch_size=20000)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
        
    #####################
    # Testing
    #####################
    def test_start(self):
        pass

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        pass

    def test_end(self, outputs):
        pass
