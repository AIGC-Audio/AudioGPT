import os
import sys
import traceback
import types
from functools import wraps
from itertools import chain
import numpy as np
import torch.utils.data
from torch.utils.data import ConcatDataset
from text_to_speech.utils.commons.hparams import hparams


def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)


def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
        indices, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, distributed=False
):
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
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

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


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader


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
        return min(self._sizes[index], hparams['max_frames'])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))


class BaseConcatDataset(ConcatDataset):
    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if not hasattr(self, 'sizes'):
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes

    def size(self, index):
        return min(self._sizes[index], hparams['max_frames'])

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.datasets[0].shuffle:
            indices = np.random.permutation(len(self))
            if self.datasets[0].sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return self.datasets[0].num_workers
