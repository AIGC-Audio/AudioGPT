import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from text_to_speech.utils.commons.dataset_utils import BaseDataset, collate_1d, collate_2d
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.commons.indexed_datasets import IndexedDataset


class EndlessDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = [i for _ in range(1000) for i in torch.randperm(
                len(self.dataset), generator=g).tolist()]
        else:
            indices = [i for _ in range(1000) for i in list(range(len(self.dataset)))]
        indices = indices[:len(indices) // self.num_replicas * self.num_replicas]
        indices = indices[self.rank::self.num_replicas]
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class VocoderDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.hparams = hparams
        self.prefix = prefix
        self.data_dir = hparams['binary_data_dir']
        self.is_infer = prefix == 'test'
        self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.hop_size = hparams['hop_size']
        self.indexed_ds = None
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.avail_idxs = [idx for idx, s in enumerate(self.sizes) if s > self.batch_max_frames]
        print(f"| {len(self.sizes) - len(self.avail_idxs)} short items are skipped in {prefix} set.")
        self.sizes = [s for idx, s in enumerate(self.sizes) if s > self.batch_max_frames]

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        item = self.indexed_ds[index]
        return item

    def __getitem__(self, index):
        index = self.avail_idxs[index]
        item = self._get_item(index)
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "mel": torch.FloatTensor(item['mel']),
            "wav": torch.FloatTensor(item['wav'].astype(np.float32)),
            "pitch": torch.LongTensor(item['pitch']),
            "f0": torch.FloatTensor(item['f0'])
        }
        return sample

    def collater(self, batch):
        if len(batch) == 0:
            return {}

        y_batch, c_batch, p_batch, f0_batch = [], [], [], []
        item_name = []
        for idx in range(len(batch)):
            item_name.append(batch[idx]['item_name'])
            x, c = batch[idx]['wav'], batch[idx]['mel']
            p, f0 = batch[idx]['pitch'], batch[idx]['f0']
            self._assert_ready_for_upsampling(x, c, self.hop_size)
            if len(c) > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                batch_max_frames = self.batch_max_frames if self.batch_max_frames != 0 else len(c) - 1
                batch_max_steps = batch_max_frames * self.hop_size
                interval_start = 0
                interval_end = len(c) - batch_max_frames
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step: start_step + batch_max_steps]
                c = c[start_frame: start_frame + batch_max_frames]
                p = p[start_frame: start_frame + batch_max_frames]
                f0 = f0[start_frame: start_frame + batch_max_frames]
                self._assert_ready_for_upsampling(y, c, self.hop_size)
            else:
                print(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [c]  # [(T' C), (T' C), ...]
            p_batch += [p]  # [(T' C), (T' C), ...]
            f0_batch += [f0]  # [(T' C), (T' C), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = collate_2d(y_batch, 0).transpose(2, 1)  # (B, 1, T)
        c_batch = collate_2d(c_batch, 0).transpose(2, 1)  # (B, C, T')
        p_batch = collate_1d(p_batch, 0)  # (B, T')
        f0_batch = collate_1d(f0_batch, 0)  # (B, T')

        # make input noise signal batch tensor
        z_batch = torch.randn(y_batch.size())  # (B, 1, T)
        return {
            'z': z_batch,
            'mels': c_batch,
            'wavs': y_batch,
            'pitches': p_batch,
            'f0': f0_batch,
            'item_name': item_name
        }

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c)) * hop_size
