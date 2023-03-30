import torch
import numpy as np
import pickle as pkl
import os, sys
import math, random
from torch.utils.data import Dataset, DataLoader
import tqdm

from audio_to_face.utils.commons.hparams import hparams
from audio_to_face.utils.commons.tensor_utils import convert_to_tensor
from audio_to_face.data_util.face3d_helper import Face3DHelper

from audio_to_face.utils.commons.indexed_datasets import IndexedDataset
from audio_to_face.tasks.audio2motion.dataset_utils.euler2quaterion import euler2quaterion, quaterion2euler

class LRS3SeqDataset(Dataset):
    def __init__(self, prefix='train'):
        self.db_key = prefix
        self.ds_path = hparams['binary_data_dir']
        self.ds = None
        self.sizes = None
        self.ordered_indices()
        self.memory_cache = {} # we use hash table to accelerate indexing
        self.face3d_helper = Face3DHelper('deep_3drecon/BFM')
        self.x_multiply = 8
        if hparams['load_db_to_memory']:
            self.load_db_to_memory()
        
    @property
    def _sizes(self):
        return self.sizes
    
    def __len__(self):
        return len(self._sizes)

    def _cal_avatar_style_encoding(self, exp, pose):
        diff_exp = exp[:-1, :] - exp[1:, :]
        exp_std = (torch.std(exp, dim = 0) - self.exp_std_mean) / self.exp_std_std
        diff_exp_std = (torch.std(diff_exp, dim = 0) - self.exp_diff_std_mean) / self.exp_diff_std_std

        diff_pose = pose[:-1, :] - pose[1:, :]
        diff_pose_std = (torch.std(diff_pose, dim = 0) - self.pose_diff_std_mean) / self.pose_diff_std_std

        return torch.cat((exp_std, diff_exp_std, diff_pose_std)) # [135,]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        sizes_fname = os.path.join(self.ds_path, f"sizes_{self.db_key}.npy")
        if os.path.exists(sizes_fname):
            sizes = np.load(sizes_fname, allow_pickle=True)
            self.sizes = sizes
        if self.sizes is None:
            self.sizes = []
            print("Counting the size of each item in dataset...")
            ds = IndexedDataset(f"{self.ds_path}/{self.db_key}")
            for i_sample in tqdm.trange(len(ds)):
                sample = ds[i_sample]
                if sample is None:
                    size = 0
                else:
                    x = sample['mel']
                    size = x.shape[-1] # time step in audio
                self.sizes.append(size)
            np.save(sizes_fname, self.sizes)
        indices = np.arange(len(self))
        indices = indices[np.argsort(np.array(self.sizes)[indices], kind='mergesort')]
        return indices

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1):
        """
        Yield mini-batches of indices bucketed by size. Batches may contain
        sequences of different lengths.

        Args:
            indices (List[int]): ordered list of dataset indices
            num_tokens_fn (callable): function that returns the number of tokens at
                a given index
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
        """
        def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            if len(batch) == 0:
                return 0
            if len(batch) == max_sentences:
                return 1
            if num_tokens > max_tokens:
                return 1
            return 0

        num_tokens_fn = lambda x: self.sizes[x]
        max_tokens = max_tokens if max_tokens is not None else 60000
        max_sentences = max_sentences if max_sentences is not None else 512
        bsz_mult = required_batch_size_multiple

        sample_len = 0
        sample_lens = []
        batch = []
        batches = []
        for i in range(len(indices)):
            idx = indices[i]
            num_tokens = num_tokens_fn(idx)
            sample_lens.append(num_tokens)
            sample_len = max(sample_len, num_tokens)

            assert sample_len <= max_tokens, (
                "sentence at index {} of size {} exceeds max_tokens "
                "limit of {}!".format(idx, sample_len, max_tokens)
            )
            num_tokens = (len(batch) + 1) * sample_len

            if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
                mod_len = max(
                    bsz_mult * (len(batch) // bsz_mult),
                    len(batch) % bsz_mult,
                )
                batches.append(batch[:mod_len])
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
            batch.append(idx)
        if len(batch) > 0:
            batches.append(batch)
        return batches

    def decode_pose(self, pose):
        """
        pose [B, T, C=7=4+3]
        """
        b,t,_ = pose.shape
        if self.normalize_target:
            pose = pose * self.pose_std + self.pose_mean
        translations = pose[:, :, :3].cpu().numpy() # [B, T, 3]
        angles = pose[:, :, 3:].cpu().numpy() # [B, T, 4]
        angles = quaterion2euler(angles.reshape([b*t,4])) # [B*T, 3]
        angles = angles.reshape([b,t,3])
        return angles, translations

    def load_db_to_memory(self):
        for idx in tqdm.trange(len(self), desc='Loading database to memory...'):
            raw_item = self._get_item(idx)
            if raw_item is None:
                continue
            item = {}
            item_id = raw_item['item_id'] # str: "<speakerID>_<clipID>"
            item['item_id'] = item_id
            # audio-related features
            mel = raw_item['mel']
            hubert = raw_item['hubert']
            item['mel'] = torch.from_numpy(mel).float() # [T_x, c=80]
            item['hubert'] = torch.from_numpy(hubert).float() # [T_x, c=80]
            if 'f0' in raw_item.keys():
                f0 = raw_item['f0']
                item['f0'] = torch.from_numpy(f0).float() # [T_x,]
            # video-related features
            coeff = raw_item['coeff'] # [T_y ~= T_x//2, c=257]
            exp = coeff[:, 80:144] 
            item['exp'] = torch.from_numpy(exp).float() # [T_y, c=64]
            translation = coeff[:, 254:257] # [T_y, c=3]
            angles = euler2quaterion(coeff[:, 224:227]) # # [T_y, c=4]
            pose = np.concatenate([translation, angles], axis=1)
            item['pose'] = torch.from_numpy(pose).float() # [T_y, c=4+3]

            # Load identity for landmark construction
            item['identity'] = torch.from_numpy(raw_item['coeff'][..., :80]).float()
            
            # Load lm3d
            t_lm, dim_lm, _ = raw_item['idexp_lm3d'].shape # [T, 68, 3]
            item['idexp_lm3d'] = torch.from_numpy(raw_item['idexp_lm3d']).reshape(t_lm, -1).float()
            eye_idexp_lm3d, mouth_idexp_lm3d = self.face3d_helper.get_eye_mouth_lm_from_lm3d(raw_item['idexp_lm3d'])
            item['eye_idexp_lm3d'] = convert_to_tensor(eye_idexp_lm3d).reshape(t_lm, -1).float()
            item['mouth_idexp_lm3d'] = convert_to_tensor(mouth_idexp_lm3d).reshape(t_lm, -1).float()
            item['ref_mean_lm3d'] = item['idexp_lm3d'].mean(dim=0).reshape([204,])
            
            self.memory_cache[idx] = item

    def _get_item(self, index):
        """
        This func is necessary to open files in multi-threads!
        """
        if self.ds is None:
            self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        return self.ds[index]
    
    def __getitem__(self, idx):
        if hparams['load_db_to_memory']:
            return self.memory_cache[idx]
        
        raw_item = self._get_item(idx)
        if raw_item is None:
            print("loading from binary data failed!")
            return None
        item = {}
        item_id = raw_item['item_id'] # str: "<speakerID>_<clipID>"
        item['item_id'] = item_id
        # audio-related features
        mel = raw_item['mel']
        hubert = raw_item['hubert']
        item['mel'] = torch.from_numpy(mel).float() # [T_x, c=80]
        item['hubert'] = torch.from_numpy(hubert).float() # [T_x, c=80]
        if 'f0' in raw_item.keys():
            f0 = raw_item['f0']
            item['f0'] = torch.from_numpy(f0).float() # [T_x,]
        # video-related features
        coeff = raw_item['coeff'] # [T_y ~= T_x//2, c=257]
        exp = coeff[:, 80:144] 
        item['exp'] = torch.from_numpy(exp).float() # [T_y, c=64]
        translation = coeff[:, 254:257] # [T_y, c=3]
        angles = euler2quaterion(coeff[:, 224:227]) # # [T_y, c=4]
        pose = np.concatenate([translation, angles], axis=1)
        item['pose'] = torch.from_numpy(pose).float() # [T_y, c=4+3]

        # Load identity for landmark construction
        item['identity'] = torch.from_numpy(raw_item['coeff'][..., :80]).float()
        
        # Load lm3d
        t_lm, dim_lm, _ = raw_item['idexp_lm3d'].shape # [T, 68, 3]
        item['idexp_lm3d'] = torch.from_numpy(raw_item['idexp_lm3d']).reshape(t_lm, -1).float()
        eye_idexp_lm3d, mouth_idexp_lm3d = self.face3d_helper.get_eye_mouth_lm_from_lm3d(raw_item['idexp_lm3d'])
        item['eye_idexp_lm3d'] = convert_to_tensor(eye_idexp_lm3d).reshape(t_lm, -1).float()
        item['mouth_idexp_lm3d'] = convert_to_tensor(mouth_idexp_lm3d).reshape(t_lm, -1).float()
        
        # item = self.memory_cache[idx]
        item['ref_mean_lm3d'] = item['idexp_lm3d'].mean(dim=0).reshape([204,])
        return item

    @staticmethod
    def _collate_2d(values, max_len=None, pad_value=0):
        """
        Convert a list of 2d tensors into a padded 3d tensor.
            values: list of Batch tensors with shape [T, C]
            return: [B, T, C]
        """
        max_len = max(v.size(0) for v in values) if max_len is None else max_len
        hidden_dim = values[0].size(1)
        batch_size = len(values)
        ret = torch.ones([batch_size,  max_len, hidden_dim],dtype=values[0].dtype) * pad_value
        for i, v in enumerate(values):
            ret[i, :v.shape[0], :].copy_(v)
        return ret

    def collater(self, samples):
        none_idx = []
        for i in range(len(samples)):
            if samples[i] is None:
                none_idx.append(i)
        for i in sorted(none_idx, reverse=True):
            del samples[i]
        if len(samples) == 0:
            return None
        batch = {}
        item_names = [s['item_id'] for s in samples]
        # style_batch = torch.stack([s["style"] for s in samples], dim=0) # [b, 135]
        x_len = max(s['mel'].size(0) for s in samples)
        x_len = x_len + (self.x_multiply - (x_len % self.x_multiply)) % self.x_multiply
        y_len = x_len // 2
        mel_batch = self._collate_2d([s["mel"] for s in samples], max_len=x_len, pad_value=0) # [b, t_max_y, 64]
        hubert_batch = self._collate_2d([s["hubert"] for s in samples], max_len=x_len, pad_value=0) # [b, t_max_y, 64]
        exp_batch = self._collate_2d([s["exp"] for s in samples], max_len=y_len, pad_value=0) # [b, t_max_y, 64]
        pose_batch = self._collate_2d([s["pose"] for s in samples], max_len=y_len, pad_value=0) # [b, t_max_y, 64]
        
        idexp_lm3d = self._collate_2d([s["idexp_lm3d"] for s in samples], max_len=y_len, pad_value=0) # [b, t_max, 1]
        ref_mean_lm3d = torch.stack([s['ref_mean_lm3d'] for s in samples], dim=0) # [b, h=204*5]
        mouth_idexp_lm3d = self._collate_2d([s["mouth_idexp_lm3d"] for s in samples], max_len=y_len, pad_value=0) # [b, t_max, 1]

        x_mask = (mel_batch.abs().sum(dim=-1) > 0).float() # [b, t_max_x]
        y_mask = (pose_batch.abs().sum(dim=-1) > 0).float() # [b, t_max_y]

        batch.update({
            'item_id': item_names,
            # 'style': style_batch,
            'mel': mel_batch,
            'hubert': hubert_batch,
            'x_mask': x_mask,
            'exp': exp_batch,
            'pose': pose_batch,
            'y_mask': y_mask,
            'idexp_lm3d': idexp_lm3d,
            'ref_mean_lm3d': ref_mean_lm3d,
            'mouth_idexp_lm3d': mouth_idexp_lm3d,
        })

        if 'f0' in samples[0].keys():
            f0_batch = self._collate_2d([s["f0"].reshape([-1,1]) for s in samples], max_len=x_len, pad_value=0).squeeze(-1) # [b, t_max_y]
            batch['f0'] = f0_batch
        return batch

    def get_dataloader(self):
        shuffle = True if self.db_key == 'train' else False
        max_tokens = 60000
        batches_idx = self.batch_by_size(self.ordered_indices(), max_tokens=max_tokens)
        batches_idx = batches_idx * 50
        random.shuffle(batches_idx)
        loader = DataLoader(self, pin_memory=True,collate_fn=self.collater, batch_sampler=batches_idx, num_workers=4)
        return loader


if __name__ == '__main__':
    from audio_to_face.utils.commons.hparams import set_hparams
    set_hparams()
    ds = LRS3SeqDataset('train')
    ret = ds[0]
    loader = ds.get_dataloader()
    pbar = tqdm.tqdm(total=len(ds.batch_by_size(ds.ordered_indices())))
    # for i in tqdm.trange(len(ds)):
    #     ds[i]
    for batch in loader:
        pbar.update(1)
    
    pbar = tqdm.tqdm(total=len(ds.batch_by_size(ds.ordered_indices())))
    for batch in loader:
        pbar.update(1)