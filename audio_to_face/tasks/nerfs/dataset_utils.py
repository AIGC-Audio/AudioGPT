import os
import tqdm
import torch
import numpy as np
from audio_to_face.utils.commons.hparams import hparams, set_hparams
from audio_to_face.utils.commons.tensor_utils import convert_to_tensor
from audio_to_face.utils.commons.image_utils import load_image_as_uint8_tensor


class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, data_dir=None, cond_type=None):
        super().__init__()
        self.data_dir = os.path.join(hparams['binary_data_dir'], hparams['video_id']) if data_dir is None else data_dir
        self.cond_type = hparams['cond_type'] if cond_type is None else cond_type
        binary_file_name = os.path.join(self.data_dir, "trainval_dataset.npy")
        ds_dict = np.load(binary_file_name, allow_pickle=True).tolist()
        if prefix == 'train':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']]
        elif prefix == 'val':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        elif prefix == 'trainval':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']] + [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        else:
            raise ValueError("prefix should in train/val !")
        self.prefix = prefix
        self.H = ds_dict['H']
        self.W = ds_dict['W']
        self.focal = ds_dict['focal']
        self.cx = ds_dict['cx']
        self.cy = ds_dict['cy']
        self.near = hparams['near'] # follow AD-NeRF, we dont use near-far in ds_dict
        self.far = hparams['far'] # follow AD-NeRF, we dont use near-far in ds_dict
        self.bg_img = torch.from_numpy(ds_dict['bg_img']).float() / 255.
        self.idexp_lm3d_mean = torch.from_numpy(ds_dict['idexp_lm3d_mean']).float()
        self.idexp_lm3d_std = torch.from_numpy(ds_dict['idexp_lm3d_std']).float()
        self.max_t = len(ds_dict['train_samples']) + len(ds_dict['val_samples'])

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        if hparams.get("load_imgs_to_memory", True):
            # disable it to save memory usage.
            # for 5500 images, it takes 1 minutes to imread, by contrast, only 1s is needed to index them in memory. 
            # But it reuqires 15GB memory for caching 5500 images at 512x512 resolution.
            if 'head_img' not in self.samples[idx].keys():
                self.samples[idx]['head_img'] = load_image_as_uint8_tensor(self.samples[idx]['head_img_fname'])
                self.samples[idx]['gt_img'] = load_image_as_uint8_tensor(self.samples[idx]['gt_img_fname'])
            head_img = self.samples[idx]['head_img']
            gt_img = self.samples[idx]['gt_img']
        else:
            head_img = load_image_as_uint8_tensor(self.samples[idx]['head_img_fname'])
            gt_img = load_image_as_uint8_tensor(self.samples[idx]['gt_img_fname'])

        sample = {
            'H': self.H,
            'W': self.W,
            'focal': self.focal,
            'cx': self.cx,
            'cy': self.cy,
            'near': self.near,
            'far': self.far,
            'idx': raw_sample['idx'],
            'rect': raw_sample['face_rect'],
            'bg_img': self.bg_img,
            'c2w': raw_sample['c2w'][:3],
            'euler': raw_sample['euler'],
            'trans': raw_sample['trans'],
            'euler_t0': self.samples[0]['euler'],
            'trans_t0': self.samples[0]['trans'],
            'c2w_t': raw_sample['c2w'][:3],
            'c2w_t0': self.samples[0]['c2w'][:3],
            't': torch.tensor([idx]).float()/ self.max_t,
        }

        sample.update({
            'head_img': head_img.float() / 255.,
            'gt_img': gt_img.float() / 255.,
        })
               
        if self.cond_type == 'deepspeech':
            sample.update({
                'cond_win': raw_sample['deepspeech_win'].unsqueeze(0), # [B=1, T=16, C=29]
                'cond_wins': raw_sample['deepspeech_wins'], # [Win=8, T=16, C=29]
            })
        elif self.cond_type == 'idexp_lm3d_normalized':
            sample['cond'] = raw_sample['idexp_lm3d_normalized'].reshape([1,-1]) # [1, 204]
            sample['cond_win'] = raw_sample['idexp_lm3d_normalized_win'].reshape([1, hparams['cond_win_size'],-1]) # [1, T_win, 204]
            sample['cond_wins'] = raw_sample['idexp_lm3d_normalized_wins'].reshape([hparams['smo_win_size'], hparams['cond_win_size'],-1]) # [smo_win, T_win, 204]
            
            if hparams.get("use_hubert", False):
                sample['hubert_win'] = raw_sample['hubert_win'].unsqueeze(0) # [Win=8, C=64]
                sample['hubert_wins'] = raw_sample['hubert_wins'].unsqueeze(0) # [Win=8, C=64]
            sample.update({
                'deepspeech_win': raw_sample['deepspeech_win'].unsqueeze(0), # [B=1, T=16, C=29]
                'deepspeech_wins': raw_sample['deepspeech_wins'], # [Win=8, T=16, C=29]
            })
        else:
            raise NotImplementedError
        
        return sample
    
    def __len__(self):
        return len(self.samples)

    def collater(self, samples):
        assert len(samples) == 1 # NeRF only take 1 image for each iteration
        return samples[0]
 

if __name__ == '__main__':
    set_hparams()
    ds = NeRFDataset('train', data_dir='data/binary/videos/May')
    ds[0]
    print("done!")