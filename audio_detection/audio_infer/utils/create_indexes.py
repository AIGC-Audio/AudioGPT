import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from utilities import create_folder, get_sub_filepaths
import config


def create_indexes(args):
    """Create indexes a for dataloader to read for training. When users have 
    a new task and their own data, they need to create similar indexes. The 
    indexes contain meta information of "where to find the data for training".
    """

    # Arguments & parameters
    waveforms_hdf5_path = args.waveforms_hdf5_path
    indexes_hdf5_path = args.indexes_hdf5_path

    # Paths
    create_folder(os.path.dirname(indexes_hdf5_path))

    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], dtype=np.bool)
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)

    print('Write to {}'.format(indexes_hdf5_path))
          

def combine_full_indexes(args):
    """Combine all balanced and unbalanced indexes hdf5s to a single hdf5. This 
    combined indexes hdf5 is used for training with full data (~20k balanced 
    audio clips + ~1.9m unbalanced audio clips).
    """

    # Arguments & parameters
    indexes_hdf5s_dir = args.indexes_hdf5s_dir
    full_indexes_hdf5_path = args.full_indexes_hdf5_path

    classes_num = config.classes_num

    # Paths
    paths = get_sub_filepaths(indexes_hdf5s_dir)
    paths = [path for path in paths if (
        'train' in path and 'full_train' not in path and 'mini' not in path)]

    print('Total {} hdf5 to combine.'.format(len(paths)))

    with h5py.File(full_indexes_hdf5_path, 'w') as full_hf:
        full_hf.create_dataset(
            name='audio_name', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S20')
        
        full_hf.create_dataset(
            name='target', 
            shape=(0, classes_num), 
            maxshape=(None, classes_num), 
            dtype=np.bool)

        full_hf.create_dataset(
            name='hdf5_path', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S200')

        full_hf.create_dataset(
            name='index_in_hdf5', 
            shape=(0,), 
            maxshape=(None,), 
            dtype=np.int32)

        for path in paths:
            with h5py.File(path, 'r') as part_hf:
                print(path)
                n = len(full_hf['audio_name'][:])
                new_n = n + len(part_hf['audio_name'][:])

                full_hf['audio_name'].resize((new_n,))
                full_hf['audio_name'][n : new_n] = part_hf['audio_name'][:]

                full_hf['target'].resize((new_n, classes_num))
                full_hf['target'][n : new_n] = part_hf['target'][:]

                full_hf['hdf5_path'].resize((new_n,))
                full_hf['hdf5_path'][n : new_n] = part_hf['hdf5_path'][:]

                full_hf['index_in_hdf5'].resize((new_n,))
                full_hf['index_in_hdf5'][n : new_n] = part_hf['index_in_hdf5'][:]
                
    print('Write combined full hdf5 to {}'.format(full_indexes_hdf5_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path of packed waveforms hdf5.')
    parser_create_indexes.add_argument('--indexes_hdf5_path', type=str, required=True, help='Path to write out indexes hdf5.')

    parser_combine_full_indexes = subparsers.add_parser('combine_full_indexes')
    parser_combine_full_indexes.add_argument('--indexes_hdf5s_dir', type=str, required=True, help='Directory containing indexes hdf5s to be combined.')
    parser_combine_full_indexes.add_argument('--full_indexes_hdf5_path', type=str, required=True, help='Path to write out full indexes hdf5 file.')

    args = parser.parse_args()
    
    if args.mode == 'create_indexes':
        create_indexes(args)

    elif args.mode == 'combine_full_indexes':
        combine_full_indexes(args)

    else:
        raise Exception('Incorrect arguments!')