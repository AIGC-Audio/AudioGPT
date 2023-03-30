import os
import tqdm
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from audio_to_face.utils.commons.hparams import hparams, set_hparams
from audio_to_face.utils.commons.tensor_utils import convert_to_tensor
from audio_to_face.utils.commons.euler2rot import euler_trans_2_c2w, c2w_to_euler_trans


class Audio2PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None):
        super().__init__()
        self.data_dir = os.path.join(hparams['binary_data_dir'], hparams['video_id']) if data_dir is None else data_dir
        binary_file_name = os.path.join(self.data_dir, "trainval_dataset.npy")
        ds_dict = np.load(binary_file_name, allow_pickle=True).tolist()
        self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']] + [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        self.num_samples = len(ds_dict['train_samples']) + len(ds_dict['val_samples'])
        self.audio_lst = [None] * self.num_samples
        self.pose_lst = [None] * self.num_samples
        self.euler_lst = [None] * self.num_samples
        self.trans_lst = [None] * self.num_samples
        for i in range(self.num_samples):
            sample = self.samples[i]
            # audio_win_size = sample['hubert_win'].shape[0]
            # audio = sample['hubert_win'][audio_win_size//2-1:audio_win_size//2+1].reshape([2*1024])
            audio = sample['deepspeech_win'][7:9,:].reshape([2*29])
            self.audio_lst[i] = audio
            self.euler_lst[i] = sample['euler']
            self.trans_lst[i] = sample['trans']
        # todo: 计算mean trans
        self.mean_trans = torch.stack(self.trans_lst).mean(dim=0)
        self.trans_lst = [self.trans_lst[i]-self.mean_trans for i in range(self.num_samples)]
        self.pose_lst = [torch.cat([self.euler_lst[i], self.trans_lst[i]], dim=-1) for i in range(self.num_samples)]
        self.pose_velocity_lst = [torch.zeros_like(self.pose_lst[0])]+[self.pose_lst[i+1] - self.pose_lst[i] for i in range(0,self.num_samples-1)]

        self.audio_lst = torch.stack(self.audio_lst)
        self.pose_lst = torch.stack(self.pose_lst)
        self.pose_velocity_lst = torch.stack(self.pose_velocity_lst)
        # self.reception_field = 30
        self.reception_field = hparams['reception_field']
        self.target_length = 5

    def __getitem__(self, idx):
        if idx < self.reception_field or idx > self.num_samples - self.target_length:
            idx = random.randint(self.reception_field, self.num_samples - self.target_length)
        sample = {
            'idx': idx,
            'audio': self.audio_lst[idx-self.reception_field: idx], # [t=30, c=512]
            'history_pose': self.pose_lst[idx-self.reception_field: idx], # [t=30, c=6]
            'history_velocity': self.pose_velocity_lst[idx-self.reception_field: idx], # [t=30, c=6]
            'target_pose': self.pose_lst[idx], # [c=6]
            'target_velocity':  self.pose_velocity_lst[idx] # [c=6]
        }
        sample['history_pose_and_velocity'] = torch.cat([sample['history_pose'], sample['history_velocity']], dim=-1) # [t=30, c=12]
        sample['target_pose_and_velocity'] = torch.cat([sample['target_pose'], sample['target_velocity']], dim=-1) # [c=12]
        return sample
    
    def __len__(self):
        return len(self.samples)

    def collater(self, samples):
        batch = {
            'idx' : [s['idx'] for s in samples],
            'audio_window': torch.stack([s['audio'] for s in samples]), # [b, t=30, c=512]
            'history_pose_and_velocity': torch.stack([s['history_pose_and_velocity'] for s in samples]), # [b, t=30, t=12]
            'target_pose_and_velocity': torch.stack([s['target_pose_and_velocity'] for s in samples]), # [b, t=12]
        }
        return batch

    def get_dataloader(self, batch_size=64):
        loader = DataLoader(self,batch_size=batch_size, pin_memory=False,collate_fn=self.collater, shuffle=True, num_workers=4)
        return loader

if __name__ == '__main__':
    set_hparams()
    ds = Audio2PoseDataset(data_dir='data/binary/videos/May')
    dl = ds.get_dataloader()
    for batch in dl:
        print("")
    print("done!")