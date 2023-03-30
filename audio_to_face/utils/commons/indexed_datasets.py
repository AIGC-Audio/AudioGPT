import pickle
from bisect import bisect
from copy import deepcopy
import numpy as np
import gzip


def int2bytes(i: int, *, signed: bool = False) -> bytes:
    length = ((i + ((i * signed) < 0)).bit_length() + 7 + signed) // 8
    return i.to_bytes(length, byteorder='little', signed=signed)


def bytes2int(b: bytes, *, signed: bool = False) -> int:
    return int.from_bytes(b, byteorder='little', signed=signed)


def load_index_data(data_file):
    index_data_size = bytes2int(data_file.read(32))
    index_data = data_file.read(index_data_size)
    index_data = pickle.loads(index_data)
    data_offsets = deepcopy(index_data['offsets'])
    id2pos = deepcopy(index_data.get('id2pos', {}))
    meta = deepcopy(index_data.get('meta', {}))
    return data_offsets, id2pos, meta


class IndexedDataset:
    def __init__(self, path, unpickle=True):
        self.path = path
        self.root_data_file = open(f"{path}.data", 'rb', buffering=-1)
        try:
            self.byte_offsets, self.id2pos, self.meta = load_index_data(self.root_data_file)
            self.data_files = [self.root_data_file]
        except:
            self.__init__old(path)
            self.meta = {}
        self.gzip = self.meta.get('gzip', False)
        if 'chunk_begin' not in self.meta:
            self.meta['chunk_begin'] = [0]
        for i in range(len(self.meta['chunk_begin'][1:])):
            self.data_files.append(open(f"{self.path}.{i + 1}.data", 'rb'))
        self.unpickle = unpickle

    def __init__old(self, path):
        self.path = path
        index_data = np.load(f"{path}.idx", allow_pickle=True).item()
        self.byte_offsets = index_data['offsets']
        self.id2pos = index_data.get('id2pos', {})
        self.data_files = [open(f"{path}.data", 'rb', buffering=-1)]

    def __getitem__(self, i):
        if self.id2pos is not None and len(self.id2pos) > 0:
            i = self.id2pos[i]
        self.check_index(i)
        
        chunk_id = bisect(self.meta['chunk_begin'][1:], self.byte_offsets[i])
        data_file = open(f"{self.path}.data", 'rb', buffering=-1)
        data_file.seek(self.byte_offsets[i] - self.meta['chunk_begin'][chunk_id])
        b = data_file.read(self.byte_offsets[i + 1] - self.byte_offsets[i])
        data_file.close()
        
        # chunk_id = bisect(self.meta['chunk_begin'][1:], self.byte_offsets[i])
        # data_file = self.data_files[chunk_id]
        # data_file.seek(self.byte_offsets[i] - self.meta['chunk_begin'][chunk_id])
        # b = data_file.read(self.byte_offsets[i + 1] - self.byte_offsets[i])

        unpickle = self.unpickle
        if unpickle:
            if self.gzip:
                b = gzip.decompress(b)
            item = pickle.loads(b)
        else:
            item = b
        return item

    def __del__(self):
        for data_file in self.data_files:
            data_file.close()

    def check_index(self, i):
        if i < 0 or i >= len(self.byte_offsets) - 1:
            raise IndexError('index out of range')

    def __len__(self):
        return len(self.byte_offsets) - 1

    def __iter__(self):
        self.iter_i = 0
        return self

    def __next__(self):
        if self.iter_i == len(self):
            raise StopIteration
        else:
            item = self[self.iter_i]
            self.iter_i += 1
            return item


class IndexedDatasetBuilder:
    def __init__(self, path, append=False, max_size=1024 * 1024 * 1024 * 64,
                 default_idx_size=1024 * 1024 * 16, gzip=False):
        self.path = self.root_path = path
        self.default_idx_size = default_idx_size
        if append:
            self.data_file = open(f"{path}.data", 'r+b')
            self.data_file.seek(0)
            self.byte_offsets, self.id2pos, self.meta = load_index_data(self.data_file)
            self.data_file.seek(0)
            self.data_file.write(bytes(default_idx_size))
            self.data_file.seek(self.byte_offsets[-1])
            self.gzip = self.meta['gzip']
        else:
            self.data_file = open(f"{path}.data", 'wb')
            self.data_file.seek(default_idx_size)
            self.byte_offsets = [default_idx_size]
            self.id2pos = {}
            self.meta = {}
            self.meta['chunk_begin'] = [0]
            self.gzip = self.meta['gzip'] = gzip
        self.root_data_file = self.data_file
        self.max_size = max_size
        self.data_chunk_id = 0

    def add_item(self, item, id=None, use_pickle=True):
        if self.byte_offsets[-1] > self.meta['chunk_begin'][-1] + self.max_size:
            if self.data_file != self.root_data_file:
                self.data_file.close()
            self.data_chunk_id += 1
            self.data_file = open(f"{self.path}.{self.data_chunk_id}.data", 'wb')
            self.data_file.seek(0)
            self.meta['chunk_begin'].append(self.byte_offsets[-1])
        if not use_pickle:
            s = item
        else:
            s = pickle.dumps(item)
            if self.gzip:
                s = gzip.compress(s, 1)
        bytes = self.data_file.write(s)
        if id is not None:
            self.id2pos[id] = len(self.byte_offsets) - 1
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.root_data_file.seek(0)
        s = pickle.dumps({'offsets': self.byte_offsets, 'id2pos': self.id2pos, 'meta': self.meta})
        assert len(s) < self.default_idx_size, (len(s), self.default_idx_size)
        len_bytes = int2bytes(len(s))
        self.root_data_file.write(len_bytes)
        self.root_data_file.seek(32)
        self.root_data_file.write(s)
        self.root_data_file.close()
        try:
            self.data_file.close()
        except:
            pass


if __name__ == "__main__":
    import random
    from tqdm import tqdm

    # builder = IndexedDatasetBuilder(ds_path, append=True)
    # for i in tqdm(range(size)):
    #     builder.add_item(items[i], i + size)
    # builder.finalize()
    # ds = IndexedDataset(ds_path)
    # for i in tqdm(range(1000)):
    #     idx = random.randint(size, 2 * size - 1)
    #     assert (ds[idx]['a'] == items[idx - size]['a']).all()
    #     idx = random.randint(0, size - 1)
    #     assert (ds[idx]['a'] == items[idx]['a']).all()

    ds_path = '/tmp/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path, max_size=1024 * 1024 * 40)
    builder.meta['lengths'] = [1, 2, 3]
    for i in tqdm(range(size)):
        builder.add_item(pickle.dumps(items[i]), i, use_pickle=False)
    builder.finalize()
    ds = IndexedDataset(ds_path)
    assert ds.meta['lengths'] == [1, 2, 3]
    for i in tqdm(range(1000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()

    # builder = IndexedDataset2Builder(ds_path, append=True)
    # builder.meta['lengths'] = [1, 2, 3, 5, 6, 7]
    # for i in tqdm(range(size)):
    #     builder.add_item(items[i], i + size)
    # builder.finalize()
    # ds = IndexedDataset2(ds_path)
    # assert ds.meta['lengths'] == [1, 2, 3, 5, 6, 7]
    # for i in tqdm(range(1000)):
    #     idx = random.randint(size, 2 * size - 1)
    #     assert (ds[idx]['a'] == items[idx - size]['a']).all()
    #     idx = random.randint(0, size - 1)
    #     assert (ds[idx]['a'] == items[idx]['a']).all()
