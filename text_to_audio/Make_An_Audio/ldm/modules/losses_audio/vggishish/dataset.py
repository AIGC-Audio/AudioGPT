import collections
import csv
import logging
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchvision

logger = logging.getLogger(f'main.{__name__}')


class VGGSound(torch.utils.data.Dataset):

    def __init__(self, split, specs_dir, transforms=None, splits_path='./data', meta_path='./data/vggsound.csv'):
        super().__init__()
        self.split = split
        self.specs_dir = specs_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.meta_path = meta_path

        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}

        split_clip_ids_path = os.path.join(splits_path, f'vggsound_{split}.txt')
        if not os.path.exists(split_clip_ids_path):
            self.make_split_files()
        clip_ids_with_timestamp = open(split_clip_ids_path).read().splitlines()
        clip_paths = [os.path.join(specs_dir, v + '_mel.npy') for v in clip_ids_with_timestamp]
        self.dataset = clip_paths
        # self.dataset = clip_paths[:10000]  # overfit one batch

        # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
        vid_classes = [self.video2target[Path(path).stem[:11]] for path in self.dataset]
        class2count = collections.Counter(vid_classes)
        self.class_counts = torch.tensor([class2count[cls] for cls in range(len(class2count))])

        # self.sample_weights = [len(self.dataset) / class2count[self.video2target[Path(path).stem[:11]]] for path in self.dataset]

    def __getitem__(self, idx):
        item = {}

        spec_path = self.dataset[idx]
        # 'zyTX_1BXKDE_16000_26000' -> 'zyTX_1BXKDE'
        video_name = Path(spec_path).stem[:11]

        item['input'] = np.load(spec_path)
        item['input_path'] = spec_path

        # if self.split in ['train', 'valid']:
        item['target'] = self.video2target[video_name]
        item['label'] = self.target2label[item['target']]

        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.dataset)

    def make_split_files(self):
        random.seed(1337)
        logger.info(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')
        # The downloaded videos (some went missing on YouTube and no longer available)
        available_vid_paths = sorted(glob(os.path.join(self.specs_dir, '*_mel.npy')))
        logger.info(f'The number of clips available after download: {len(available_vid_paths)}')

        # original (full) train and test sets
        vggsound_meta = list(csv.reader(open(self.meta_path), quotechar='"'))
        train_vids = {row[0] for row in vggsound_meta if row[3] == 'train'}
        test_vids = {row[0] for row in vggsound_meta if row[3] == 'test'}
        logger.info(f'The number of videos in vggsound train set: {len(train_vids)}')
        logger.info(f'The number of videos in vggsound test set: {len(test_vids)}')

        # class counts in test set. We would like to have the same distribution in valid
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        label2target = {label: target for target, label in enumerate(unique_classes)}
        video2target = {row[0]: label2target[row[2]] for row in vggsound_meta}
        test_vid_classes = [video2target[vid] for vid in test_vids]
        test_target2count = collections.Counter(test_vid_classes)

        # now given the counts from test set, sample the same count for validation and the rest leave in train
        train_vids_wo_valid, valid_vids = set(), set()
        for target, label in enumerate(label2target.keys()):
            class_train_vids = [vid for vid in train_vids if video2target[vid] == target]
            random.shuffle(class_train_vids)
            count = test_target2count[target]
            valid_vids.update(class_train_vids[:count])
            train_vids_wo_valid.update(class_train_vids[count:])

        # make file with a list of available test videos (each video should contain timestamps as well)
        train_i = valid_i = test_i = 0
        with open(os.path.join(self.splits_path, 'vggsound_train.txt'), 'w') as train_file, \
             open(os.path.join(self.splits_path, 'vggsound_valid.txt'), 'w') as valid_file, \
             open(os.path.join(self.splits_path, 'vggsound_test.txt'), 'w') as test_file:
            for path in available_vid_paths:
                path = path.replace('_mel.npy', '')
                vid_name = Path(path).name
                # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
                if vid_name[:11] in train_vids_wo_valid:
                    train_file.write(vid_name + '\n')
                    train_i += 1
                elif vid_name[:11] in valid_vids:
                    valid_file.write(vid_name + '\n')
                    valid_i += 1
                elif vid_name[:11] in test_vids:
                    test_file.write(vid_name + '\n')
                    test_i += 1
                else:
                    raise Exception(f'Clip {vid_name} is neither in train, valid nor test. Strange.')

        logger.info(f'Put {train_i} clips to the train set and saved it to ./data/vggsound_train.txt')
        logger.info(f'Put {valid_i} clips to the valid set and saved it to ./data/vggsound_valid.txt')
        logger.info(f'Put {test_i} clips to the test set and saved it to ./data/vggsound_test.txt')


if __name__ == '__main__':
    from transforms import Crop, StandardNormalizeAudio, ToTensor
    specs_path = '/home/nvme/data/vggsound/features/melspec_10s_22050hz/'

    transforms = torchvision.transforms.transforms.Compose([
        StandardNormalizeAudio(specs_path),
        ToTensor(),
        Crop([80, 848]),
    ])

    datasets = {
        'train': VGGSound('train', specs_path, transforms),
        'valid': VGGSound('valid', specs_path, transforms),
        'test': VGGSound('test', specs_path, transforms),
    }

    print(datasets['train'][0])
    print(datasets['valid'][0])
    print(datasets['test'][0])

    print(datasets['train'].class_counts)
    print(datasets['valid'].class_counts)
    print(datasets['test'].class_counts)
