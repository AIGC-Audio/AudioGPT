# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py

import collections
import sys
from loguru import logger
from pprint import pformat

import numpy as np
import pandas as pd
import scipy
import six
import sklearn.preprocessing as pre
import torch
import tqdm
import yaml

from scipy.interpolate import interp1d

def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs
    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    arguments = dict(yaml_config, **kwargs)
    return arguments


def find_contiguous_regions(activity_array): # in this part, if you cannot understand the binary operation, I think you can write a O(n) complexity method
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder
    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters
    """
    change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0] 
    change_indices += 1
    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]
    # print(change_indices.reshape((-1, 2)))
    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def split_train_cv(
        data_frame: pd.DataFrame,
        frac: float = 0.9,
        y=None,  # Only for stratified, computes necessary split
        **kwargs):
    """split_train_cv

    :param data_frame:
    :type data_frame: pd.DataFrame
    :param frac:
    :type frac: float
    """
    if kwargs.get('mode',
                  None) == 'urbansed':  # Filenames are DATA_-1 DATA_-2 etc
        data_frame.loc[:, 'id'] = data_frame.groupby(
            data_frame['filename'].str.split('_').apply(
                lambda x: '_'.join(x[:-1]))).ngroup()
        sampler = np.random.permutation(data_frame['id'].nunique())
        num_train = int(frac * len(sampler))
        train_indexes = sampler[:num_train]
        cv_indexes = sampler[num_train:]
        train_data = data_frame[data_frame['id'].isin(train_indexes)]
        cv_data = data_frame[data_frame['id'].isin(cv_indexes)]
        del train_data['id']
        del cv_data['id']
    elif kwargs.get('mode', None) == 'stratified': #  stratified --> 分层的 ?
        # Use statified sampling
        from skmultilearn.model_selection import iterative_train_test_split
        index_train, _, index_cv, _ = iterative_train_test_split(
            data_frame.index.values.reshape(-1, 1), y, test_size=1. - frac)
        train_data = data_frame[data_frame.index.isin(index_train.squeeze())]
        cv_data = data_frame[data_frame.index.isin(index_cv.squeeze())] # cv --> cross validation
    else:
        # Simply split train_test
        train_data = data_frame.sample(frac=frac, random_state=10)
        cv_data = data_frame[~data_frame.index.isin(train_data.index)]
    return train_data, cv_data



def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'): # print yaml file
    """pprint_dict
    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def getfile_outlogger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger

# according label, get encoder
def train_labelencoder(labels: pd.Series, sparse=True):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (many hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to be series"
    if isinstance(labels[0], six.string_types):
        # In case of using non processed strings, e.g., Vaccum, Speech
        label_array = labels.str.split(',').values.tolist() # split label according to ','
    elif isinstance(labels[0], np.ndarray):
        # Encoder does not like to see numpy array
        label_array = [lab.tolist() for lab in labels]
    elif isinstance(labels[0], collections.Iterable):
        label_array = labels
    encoder = pre.MultiLabelBinarizer(sparse_output=sparse)
    encoder.fit(label_array)
    return encoder


def encode_labels(labels: pd.Series, encoder=None, sparse=True):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (many hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to be series"
    instance = labels.iloc[0]
    if isinstance(instance, six.string_types):
        # In case of using non processed strings, e.g., Vaccum, Speech
        label_array = labels.str.split(',').values.tolist()
    elif isinstance(instance, np.ndarray):
        # Encoder does not like to see numpy array
        label_array = [lab.tolist() for lab in labels]
    elif isinstance(instance, collections.Iterable):
        label_array = labels
    # get label_array, it is a list ,contain a lot of label, this label are string type
    if not encoder:
        encoder = pre.MultiLabelBinarizer(sparse_output=sparse) # if we encoder is None, we should init a encoder firstly.
        encoder.fit(label_array)
    labels_encoded = encoder.transform(label_array) # transform string to digit
    return labels_encoded, encoder

    # return pd.arrays.SparseArray(
    # [row.toarray().ravel() for row in labels_encoded]), encoder


def decode_with_timestamps(events,labels: np.array):
    """decode_with_timestamps
    Decodes the predicted label array (2d) into a list of
    [(Labelname, onset, offset), ...]

    :param encoder: Encoder during training
    :type encoder: pre.MultiLabelBinarizer
    :param labels: n-dim array
    :type labels: np.array
    """
    # print('events ',events)
    # print('labels ',labels.shape)
    #assert 1==2
    if labels.ndim == 2:
        #print('...')
        return [_decode_with_timestamps(events[i],labels[i]) for i in range(labels.shape[0])]
    else:
        return _decode_with_timestamps(events,labels)


def median_filter(x, window_size, threshold=0.5):
    """median_filter
    :param x: input prediction array of shape (B, T, C) or (B, T).
        Input is a sequence of probabilities 0 <= x <= 1
    :param window_size: An integer to use 
    :param threshold: Binary thresholding threshold
    """
    x = binarize(x, threshold=threshold) # transfer to 0 or 1
    if x.ndim == 3:
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        # Assume input is class-specific median filtering
        # E.g, Batch x Time  [1, 501]
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1:
        # Assume input is standard median pooling, class-independent
        # E.g., Time x Class [501, 10]
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def _decode_with_timestamps(events,labels):
    result_labels = []
    # print('.......')
    # print('labels ',labels.shape)
    # print(labels)
    change_indices = find_contiguous_regions(labels)
    # print(change_indices)
    # assert 1==2
    for row in change_indices:
        result_labels.append((events,row[0], row[1]))
    return result_labels

def inverse_transform_labels(encoder, pred):
    if pred.ndim == 3:
        return [encoder.inverse_transform(x) for x in pred]
    else:
        return encoder.inverse_transform(pred)


def binarize(pred, threshold=0.5):
    # Batch_wise
    if pred.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in pred])
    else:
        return pre.binarize(pred, threshold=threshold)


def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays

    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)


def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True): # in nature, double_threshold considers boundary question
    """_double_threshold
    Computes a double threshold over the input array

    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0] # return the index, where value is greater than high_thres
    locations = x > low_thres # return true of false
    encoded_pairs = find_contiguous_regions(locations)
    # print('encoded_pairs ',encoded_pairs)
    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs)) # find encoded_pair where inclide a high_lacations
    #print('filtered_list ',filtered_list)
    filtered_list = connect_(filtered_list, n_connect) # if the distance of two pair is less than n_connect, we can merge them
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list


def connect_clusters(x, n=1):
    if x.ndim == 1:
        return connect_clusters_(x, n)
    if x.ndim >= 2:
        return np.apply_along_axis(lambda a: connect_clusters_(a, n=n), -2, x)


def connect_clusters_(x, n=1):
    """connect_clusters_
    Connects clustered predictions (0,1) in x with range n

    :param x: Input array. zero-one format
    :param n: Number of frames to skip until connection can be made
    """
    assert x.ndim == 1, "input needs to be 1d"
    reg = find_contiguous_regions(x)
    start_end = connect_(reg, n=n)
    zero_one_arr = np.zeros_like(x, dtype=int)
    for sl in start_end:
        zero_one_arr[sl[0]:sl[1]] = 1
    return zero_one_arr


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters 
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


def predictions_to_time(df, ratio):
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df

def upgrade_resolution(arr, scale):
    print('arr ',arr.shape)
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale
# a = [0.1,0.2,0.3,0.8,0.4,0.1,0.3,0.9,0.4]
# a = np.array(a)
# b = a>0.2
# _double_threshold(a,0.7,0.2)