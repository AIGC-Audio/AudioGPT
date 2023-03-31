import logging
import os
from pathlib import Path

import albumentations
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(f'main.{__name__}')


class StandardNormalizeAudio(object):
    '''
        Frequency-wise normalization
    '''
    def __init__(self, specs_dir, train_ids_path='./data/vggsound_train.txt', cache_path='./data/'):
        self.specs_dir = specs_dir
        self.train_ids_path = train_ids_path
        # making the stats filename to match the specs dir name
        self.cache_path = os.path.join(cache_path, f'train_means_stds_{Path(specs_dir).stem}.txt')
        logger.info('Assuming that the input stats are calculated using preprocessed spectrograms (log)')
        self.train_stats = self.calculate_or_load_stats()

    def __call__(self, item):
        # just to generalizat the input handling. Useful for FID, IS eval and training other staff
        if isinstance(item, dict):
            if 'input' in item:
                input_key = 'input'
            elif 'image' in item:
                input_key = 'image'
            else:
                raise NotImplementedError
            item[input_key] = (item[input_key] - self.train_stats['means']) / self.train_stats['stds']
        elif isinstance(item, torch.Tensor):
            # broadcasts np.ndarray (80, 1) to (1, 80, 1) because item is torch.Tensor (B, 80, T)
            item = (item - self.train_stats['means']) / self.train_stats['stds']
        else:
            raise NotImplementedError
        return item

    def calculate_or_load_stats(self):
        try:
            # (F, 2)
            train_stats = np.loadtxt(self.cache_path)
            means, stds = train_stats.T
            logger.info('Trying to load train stats for Standard Normalization of inputs')
        except OSError:
            logger.info('Could not find the precalculated stats for Standard Normalization. Calculating...')
            train_vid_ids = open(self.train_ids_path)
            specs_paths = [os.path.join(self.specs_dir, f'{i.rstrip()}_mel.npy') for i in train_vid_ids]
            means = [None] * len(specs_paths)
            stds = [None] * len(specs_paths)
            for i, path in enumerate(tqdm(specs_paths)):
                spec = np.load(path)
                means[i] = spec.mean(axis=1)
                stds[i] = spec.std(axis=1)
            # (F) <- (num_files, F)
            means = np.array(means).mean(axis=0)
            stds = np.array(stds).mean(axis=0)
            # saving in two columns
            np.savetxt(self.cache_path, np.vstack([means, stds]).T, fmt='%0.8f')
        means = means.reshape(-1, 1)
        stds = stds.reshape(-1, 1)
        return {'means': means, 'stds': stds}

class ToTensor(object):

    def __call__(self, item):
        item['input'] = torch.from_numpy(item['input']).float()
        # if 'target' in item:
        item['target'] = torch.tensor(item['target'])
        return item

class Crop(object):

    def __init__(self, cropped_shape=None, random_crop=False):
        self.cropped_shape = cropped_shape
        if cropped_shape is not None:
            mel_num, spec_len = cropped_shape
            if random_crop:
                self.cropper = albumentations.RandomCrop
            else:
                self.cropper = albumentations.CenterCrop
            self.preprocessor = albumentations.Compose([self.cropper(mel_num, spec_len)])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __call__(self, item):
        item['input'] = self.preprocessor(image=item['input'])['image']
        return item


if __name__ == '__main__':
    cropper = Crop([80, 848])
    item = {'input': torch.rand([80, 860])}
    outputs = cropper(item)
    print(outputs['input'].shape)
