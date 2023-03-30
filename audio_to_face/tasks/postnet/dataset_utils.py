import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from audio_to_face.utils.commons.hparams import hparams, set_hparams

from audio_to_face.tasks.audio2motion.dataset_utils.lrs3_dataset import LRS3SeqDataset


class PostnetDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, data_dir=None):
        super().__init__()
        self.person_binary_data_dir = os.path.join(hparams['person_binary_data_dir'], hparams['video_id']) if data_dir is None else data_dir
        binary_file_name = os.path.join(self.person_binary_data_dir, "trainval_dataset.npy")
        person_ds_dict = np.load(binary_file_name, allow_pickle=True).tolist()
        mel = person_ds_dict['mel']
        f0 = person_ds_dict['f0'].reshape([-1,1])
        hubert = person_ds_dict['hubert']
        # if len(mel.shape) == 0: # is object
        #     mel = mel.tolist()['mel']
        train_lm3d_normalized = np.stack([sample['idexp_lm3d_normalized'] for sample in person_ds_dict['train_samples']], axis=0)
        train_lm3d = np.stack([sample['idexp_lm3d'] for sample in person_ds_dict['train_samples']], axis=0)
        val_lm3d_normalized = np.stack([sample['idexp_lm3d_normalized'] for sample in person_ds_dict['val_samples']], axis=0)
        val_lm3d = np.stack([sample['idexp_lm3d'] for sample in person_ds_dict['val_samples']], axis=0)
        lm3d_len = train_lm3d_normalized.shape[0] + val_lm3d_normalized.shape[0]
        mel_len = mel.shape[0]
        if mel_len > 2 * lm3d_len:
            mel = mel[:2*lm3d_len] 
            f0 = f0[:2*lm3d_len] 
            hubert = hubert[:2*lm3d_len]
        elif mel_len < 2 * lm3d_len:
            num_to_pad = 2 * lm3d_len - mel_len
            mel = np.pad(mel, ((0,num_to_pad),(0,0)), mode="constant")
            f0 = np.pad(f0, ((0,num_to_pad),(0,0)), mode="constant")
            hubert = np.pad(hubert, ((0,num_to_pad),(0,0)), mode="constant")

        if prefix == 'train':
            lm3d_normalized = train_lm3d_normalized
            lm3d = train_lm3d
            mel = mel[:lm3d_normalized.shape[0]*2]
            f0 = f0[:lm3d_normalized.shape[0]*2]
            hubert = hubert[:lm3d_normalized.shape[0]*2]
        elif prefix == 'val':
            lm3d_normalized = val_lm3d_normalized
            lm3d = val_lm3d
            mel = mel[train_lm3d_normalized.shape[0]*2 : train_lm3d_normalized.shape[0]*2 + lm3d_normalized.shape[0]*2]
            f0 = f0[train_lm3d_normalized.shape[0]*2 : train_lm3d_normalized.shape[0]*2 + lm3d_normalized.shape[0]*2]
            hubert = hubert[train_lm3d_normalized.shape[0]*2 : train_lm3d_normalized.shape[0]*2 + lm3d_normalized.shape[0]*2]
        else:
            raise ValueError("prefix should in train/val !")
        

        target_x_len = mel.shape[0] // 8 * 8
        target_y_len = target_x_len // 2
        mel = mel[:target_x_len]
        f0 = f0[:target_x_len].reshape([-1,])
        hubert = hubert[:target_x_len]
        lm3d_normalized = lm3d_normalized[:target_y_len]
        lm3d_normalized = lm3d_normalized.reshape(lm3d_normalized.shape[0], -1)
        lm3d = lm3d[:target_y_len]
        lm3d = lm3d.reshape(lm3d_normalized.shape[0], -1)

        idexp_lm3d_mean = person_ds_dict['idexp_lm3d_mean'][:target_y_len].reshape(1, -1)
        idexp_lm3d_std = person_ds_dict['idexp_lm3d_std'][:target_y_len].reshape(1, -1)
        self.person_ds = {
            'mel': torch.from_numpy(mel).float().unsqueeze(0),
            'f0': torch.from_numpy(f0).float().unsqueeze(0),
            'hubert': torch.from_numpy(hubert).float().unsqueeze(0),
            'idexp_lm3d_normalized': torch.from_numpy(lm3d_normalized).float().unsqueeze(0),
            'idexp_lm3d': torch.from_numpy(lm3d).float().unsqueeze(0),
            'x_mask': torch.ones([target_x_len,]).float().unsqueeze(0),
            'y_mask': torch.ones([target_y_len,]).float().unsqueeze(0),
            'idexp_lm3d_mean': torch.from_numpy(idexp_lm3d_mean).float().unsqueeze(0),
            'idexp_lm3d_std': torch.from_numpy(idexp_lm3d_std).float().unsqueeze(0),
        }

        self.audio2motion_ds = LRS3SeqDataset(prefix)
        
    def __getitem__(self, idx):
        sample = self.audio2motion_ds[idx]
        return sample
    
    def __len__(self):
        return len(self.samples)

    def collater(self, samples):
        batch = self.audio2motion_ds.collater(samples)
        batch['person_ds'] = self.person_ds
        return batch

    def get_dataloader(self):
        max_tokens = 60000
        batches_idx = self.audio2motion_ds.batch_by_size(self.audio2motion_ds.ordered_indices(), max_tokens=max_tokens)
        loader = DataLoader(self, pin_memory=False,collate_fn=self.collater, batch_sampler=batches_idx, num_workers=0)
        # loader = DataLoader(self, pin_memory=False,collate_fn=self.collater, batch_sampler=batches_idx, num_workers=4)
        return loader

if __name__ == '__main__':
    set_hparams()
    ds = PostnetDataset("train")
    ds[0]
    print("done!")