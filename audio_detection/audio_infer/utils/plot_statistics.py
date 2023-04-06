import os
import sys
import numpy as np
import argparse
import h5py
import time
import _pickle as cPickle
import _pickle
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

from utilities import (create_folder, get_filename, d_prime)
import config


def _load_metrics0(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
    fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
    workspace0 = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/pub_audioset_tagging_cnn_transfer'
    statistics_path = os.path.join(workspace0, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
    bal_map = np.mean(bal_map, axis=-1)
    test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
    test_map = np.mean(test_map, axis=-1)
    legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

    # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
    return bal_map, test_map, legend


def _load_metrics0_classwise(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
    fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
    workspace0 = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/pub_audioset_tagging_cnn_transfer'
    statistics_path = os.path.join(workspace0, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    return statistics_dict['test'][300]['average_precision']


def _load_metrics0_classwise2(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
    fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
    workspace0 = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/pub_audioset_tagging_cnn_transfer'
    statistics_path = os.path.join(workspace0, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    k = 270
    mAP = np.mean(statistics_dict['test'][k]['average_precision'])
    mAUC = np.mean(statistics_dict['test'][k]['auc'])
    dprime = d_prime(mAUC)
    return mAP, mAUC, dprime


def _load_metrics_classwise(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
    fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
    workspace = '/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/cvssp/pub_audioset_tagging_cnn'
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))
    
    k = 300
    mAP = np.mean(statistics_dict['test'][k]['average_precision'])
    mAUC = np.mean(statistics_dict['test'][k]['auc'])
    dprime = d_prime(mAUC)
    return mAP, mAUC, dprime


def plot(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    select = args.select
    
    classes_num = config.classes_num
    max_plot_iteration = 1000000
    iterations = np.arange(0, max_plot_iteration, 2000)

    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')
        
    save_out_path = 'results/{}.pdf'.format(select)
    create_folder(os.path.dirname(save_out_path))
    
    # Read labels
    labels = config.labels
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
        
    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))

        bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
        bal_map = np.mean(bal_map, axis=-1)
        test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
        test_map = np.mean(test_map, axis=-1)
        legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

        # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
        return bal_map, test_map, legend
        
    bal_alpha = 0.3
    test_alpha = 1.0
    lines = []

    if select == '1_cnn13':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_no_dropout', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_no_specaug', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_no_specaug', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_no_dropout', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_no_mixup', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_mixup_in_wave', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_mixup_in_wave', color='c', alpha=test_alpha)
        lines.append(line)

    elif select == '1_pooling':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_gwrp', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_gmpgapgwrp', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_att', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_gmpgapatt', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_resnet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet18', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='ResNet18', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet34', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet50', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='resnet50', color='c', alpha=test_alpha)
        lines.append(line)

    elif select == '1_densenet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'DenseNet121', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='densenet121', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'DenseNet201', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='densenet201', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_cnn9':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn5', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn5', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn9', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn9', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_hop':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            500, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_hop500', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            640, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_hop640', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            1000, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_hop1000', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_emb':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_emb32', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_emb128', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb512', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13_emb512', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_mobilenet':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='mobilenetv1', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV2', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='mobilenetv2', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_waveform':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_LeeNet', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_LeeNet', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_LeeNet18', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_LeeNet18', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_DaiNet', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_DaiNet', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_ResNet34', color='c', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet50', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_ResNet50', color='m', alpha=test_alpha)
        lines.append(line)

    elif select == '1_waveform_cnn2d':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_SpAndWav', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_WavCnn2d', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_WavCnn2d', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_decision_level':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelMax', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_DecisionLevelMax', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelAvg', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_DecisionLevelAvg', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelAtt', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_DecisionLevelAtt', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_transformer':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_Transformer1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_Transformer1', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_Transformer3', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_Transformer3', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_Transformer6', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_Transformer6', color='k', alpha=test_alpha)
        lines.append(line)

    elif select == '1_aug':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,balanced,mixup', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'none', 'none', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,none,none', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,balanced,none', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup_from_0_epoch', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,balanced,mixup_from_0_epoch', color='m', alpha=test_alpha)
        lines.append(line)

    elif select == '1_bal_train_aug':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,balanced,mixup', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'none', 'none', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,none,none', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,balanced,none', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup_from_0_epoch', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,balanced,mixup_from_0_epoch', color='m', alpha=test_alpha)
        lines.append(line)

    elif select == '1_sr':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_16k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_16k', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_8k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_8k', color='b', alpha=test_alpha)
        lines.append(line)

    elif select == '1_time_domain':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_mixup_time_domain', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_time_domain', color='b', alpha=test_alpha)
        lines.append(line)

    elif select == '1_partial_full':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.9_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,partial_0.9', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.8_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,partial_0.8', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.7_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,partial_0.7', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.5_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,partial_0.5', color='m', alpha=test_alpha)
        lines.append(line)

    elif select == '1_window':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 2048, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_win2048', color='b', alpha=test_alpha)
        lines.append(line)

    elif select == '1_melbins':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 32, 50, 14000, 'full_train', 'Cnn14_mel32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_mel32', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 128, 50, 14000, 'full_train', 'Cnn14_mel128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_mel128', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '1_alternate':
        max_plot_iteration = 2000000
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'alternate', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_alternate', color='b', alpha=test_alpha)
        lines.append(line)

    elif select == '2_all':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn9', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn9', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn5', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn5', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='MobileNetV1', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn1d_ResNet34', color='grey', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='ResNet34', color='grey', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_WavCnn2d', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_WavCnn2d', color='m', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_SpAndWav', color='orange', alpha=test_alpha)
        lines.append(line)

    elif select == '2_emb':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_emb32', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_128', color='k', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb512', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn13_512', color='g', alpha=test_alpha)
        lines.append(line)

    elif select == '2_aug':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn13', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
                320, 64, 50, 14000, 'full_train', 'Cnn13_no_specaug', 'clip_bce', 'none', 'none', 32)
        line, = ax.plot(bal_map, color='c', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14,none,none', color='c', alpha=test_alpha)
        lines.append(line)

        

    ax.set_ylim(0, 1.)
    ax.set_xlim(0, len(iterations))
    ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=2)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=lines, bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def plot_for_paper(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    select = args.select
    
    classes_num = config.classes_num
    max_plot_iteration = 1000000
    iterations = np.arange(0, max_plot_iteration, 2000)

    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')
        
    save_out_path = 'results/paper_{}.pdf'.format(select)
    create_folder(os.path.dirname(save_out_path))
    
    # Read labels
    labels = config.labels
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    lines = []
        
    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))

        bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
        bal_map = np.mean(bal_map, axis=-1)
        test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
        test_map = np.mean(test_map, axis=-1)
        legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

        # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
        return bal_map, test_map, legend

    bal_alpha = 0.3
    test_alpha = 1.0
    lines = []
    linewidth = 1.

    max_plot_iteration = 540000

    if select == '2_all':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn9', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='cnn9', color='r', alpha=test_alpha)
        # lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn5', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='cnn5', color='g', alpha=test_alpha)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='MobileNetV1', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='Cnn1d_ResNet34', color='grey', alpha=test_alpha)
        # lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn13_WavCnn2d', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        # line, = ax.plot(test_map, label='Wavegram-CNN', color='g', alpha=test_alpha, linewidth=linewidth)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='Wavegram-Logmel-CNN', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

    elif select == '2_emb':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,emb=2048', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,emb=32', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,emb=128', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn13_emb512', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='Cnn13_512', color='g', alpha=test_alpha)
        # lines.append(line)

    elif select == '2_bal':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,bal,mixup (1.9m)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_mixup_time_domain', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='y', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,bal,mixup-wav (1.9m)', color='y', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'none', 'none', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,no-bal,no-mixup (1.9m)', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,bal,no-mixup (1.9m)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,bal,no-mixup (20k)', color='k', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,bal,mixup (20k)', color='m', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

    elif select == '2_sr':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,32kHz', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_16k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,16kHz', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_8k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,8kHz', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

    elif select == '2_partial':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14 (100% full)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'partial_0.9_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        # line, = ax.plot(test_map, label='cnn14,partial_0.9', color='b', alpha=test_alpha, linewidth=linewidth)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.8_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14 (80% full)', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'partial_0.7_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='k', alpha=bal_alpha, linewidth=linewidth)
        # line, = ax.plot(test_map, label='cnn14,partial_0.7', color='k', alpha=test_alpha, linewidth=linewidth)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.5_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='cnn14 (50% full)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

    elif select == '2_melbins':
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax.plot(test_map, label='CNN14,64-melbins', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 32, 50, 14000, 'full_train', 'Cnn14_mel32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='CNN14,32-melbins', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 128, 50, 14000, 'full_train', 'Cnn14_mel128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='CNN14,128-melbins', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, len(iterations))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('mAP')
    ax.xaxis.set_ticks(np.arange(0, len(iterations), 50))
    # ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    ax.xaxis.set_ticklabels(['0', '100k', '200k', '300k', '400k', '500k'])
    ax.yaxis.set_ticks(np.arange(0, 0.81, 0.05))
    ax.yaxis.set_ticklabels(['0', '', '0.1', '', '0.2', '', '0.3', '', '0.4', '', '0.5', '', '0.6', '', '0.7', '', '0.8'])
    # ax.yaxis.set_ticklabels(np.around(np.arange(0, 0.81, 0.05), decimals=2))        
    ax.yaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
    ax.xaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
    plt.legend(handles=lines, loc=2)
    plt.tight_layout(0, 0, 0)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=lines, bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def plot_for_paper2(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    classes_num = config.classes_num
    max_plot_iteration = 1000000
    iterations = np.arange(0, max_plot_iteration, 2000)

    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')
        
    save_out_path = 'results/paper2.pdf'
    create_folder(os.path.dirname(save_out_path))
    
    # Read labels
    labels = config.labels
    
    # Plot
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    lines = []
        
    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))

        bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
        bal_map = np.mean(bal_map, axis=-1)
        test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
        test_map = np.mean(test_map, axis=-1)
        legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

        # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
        return bal_map, test_map, legend

    def _load_metrics0(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
        workspace0 = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/pub_audioset_tagging_cnn_transfer'
        statistics_path = os.path.join(workspace0, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))

        bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
        bal_map = np.mean(bal_map, axis=-1)
        test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
        test_map = np.mean(test_map, axis=-1)
        legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

        # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
        return bal_map, test_map, legend
        
    bal_alpha = 0.3
    test_alpha = 1.0
    lines = []
    linewidth = 1.

    max_plot_iteration = 540000

    if True:
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 0].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 0].plot(test_map, label='CNN14', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn9', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='cnn9', color='r', alpha=test_alpha)
        # lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn5', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='cnn5', color='g', alpha=test_alpha)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 0].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 0].plot(test_map, label='MobileNetV1', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='Cnn1d_ResNet34', color='grey', alpha=test_alpha)
        # lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'ResNet34', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax[0, 0].plot(bal_map, color='k', alpha=bal_alpha, linewidth=linewidth)
        # line, = ax[0, 0].plot(test_map, label='ResNet38', color='k', alpha=test_alpha, linewidth=linewidth)
        # lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn13_WavCnn2d', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        # line, = ax.plot(test_map, label='Wavegram-CNN', color='g', alpha=test_alpha, linewidth=linewidth)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 0].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 0].plot(test_map, label='Wavegram-Logmel-CNN', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0, 0].legend(handles=lines, loc=2)
        ax[0, 0].set_title('(a) Comparison of architectures')

    if True:
        lines = []
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 1].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,mixup (1.9m)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'none', 'none', 32)
        line, = ax[0, 1].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,no-bal,no-mixup (1.9m)', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_mixup_time_domain', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 1].plot(bal_map, color='y', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,mixup-wav (1.9m)', color='y', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)
        line, = ax[0, 1].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,no-mixup (1.9m)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)
        line, = ax[0, 1].plot(bal_map, color='k', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,no-mixup (20k)', color='k', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 1].plot(bal_map, color='m', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,mixup (20k)', color='m', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0, 1].legend(handles=lines, loc=2, fontsize=8)

        ax[0, 1].set_title('(b) Comparison of training data and augmentation')

    if True:
        lines = []
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 2].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 2].plot(test_map, label='CNN14,emb=2048', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 2].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 2].plot(test_map, label='CNN14,emb=32', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics0('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[0, 2].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 2].plot(test_map, label='CNN14,emb=128', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0, 2].legend(handles=lines, loc=2)
        ax[0, 2].set_title('(c) Comparison of embedding size')

    if True:
        lines = []
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 0].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 0].plot(test_map, label='CNN14 (100% full)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.8_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 0].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 0].plot(test_map, label='CNN14 (80% full)', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'partial_0.5_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 0].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 0].plot(test_map, label='cnn14 (50% full)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[1, 0].legend(handles=lines, loc=2)
        ax[1, 0].set_title('(d) Comparison of amount of training data')

    if True:
        lines = []
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 1].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 1].plot(test_map, label='CNN14,32kHz', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_16k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 1].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 1].plot(test_map, label='CNN14,16kHz', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_8k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 1].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 1].plot(test_map, label='CNN14,8kHz', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[1, 1].legend(handles=lines, loc=2)
        ax[1, 1].set_title('(e) Comparison of sampling rate')

    if True:
        lines = []
        iterations = np.arange(0, max_plot_iteration, 2000)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 2].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 2].plot(test_map, label='CNN14,64-melbins', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 32, 50, 14000, 'full_train', 'Cnn14_mel32', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 2].plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax[1, 2].plot(test_map, label='CNN14,32-melbins', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 128, 50, 14000, 'full_train', 'Cnn14_mel128', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax[1, 2].plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax[1, 2].plot(test_map, label='CNN14,128-melbins', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[1, 2].legend(handles=lines, loc=2)
        ax[1, 2].set_title('(f) Comparison of mel bins number')

    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(0, 0.8)
            ax[i, j].set_xlim(0, len(iterations))
            ax[i, j].set_xlabel('Iterations')
            ax[i, j].set_ylabel('mAP')
            ax[i, j].xaxis.set_ticks(np.arange(0, len(iterations), 50))
            # ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
            ax[i, j].xaxis.set_ticklabels(['0', '100k', '200k', '300k', '400k', '500k'])
            ax[i, j].yaxis.set_ticks(np.arange(0, 0.81, 0.05))
            ax[i, j].yaxis.set_ticklabels(['0', '', '0.1', '', '0.2', '', '0.3', '', '0.4', '', '0.5', '', '0.6', '', '0.7', '', '0.8'])
            # ax.yaxis.set_ticklabels(np.around(np.arange(0, 0.81, 0.05), decimals=2))        
            ax[i, j].yaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
            ax[i, j].xaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)

    plt.tight_layout(0, 1, 0)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=lines, bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def table_values(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    select = args.select
    
    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size, iteration):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))
 
        idx = iteration // 2000
        mAP = np.mean(statistics_dict['test'][idx]['average_precision'])
        mAUC = np.mean(statistics_dict['test'][idx]['auc'])
        dprime = d_prime(mAUC)

        print('mAP: {:.3f}'.format(mAP))
        print('mAUC: {:.3f}'.format(mAUC))
        print('dprime: {:.3f}'.format(dprime))


    if select == 'cnn13':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn5':
        iteration = 440000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn5', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn9':
        iteration = 440000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn9', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_decisionlevelmax':
        iteration = 400000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelMax', 'clip_bce', 'balanced', 'mixup', 32, iteration)
    
    elif select == 'cnn13_decisionlevelavg':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelAvg', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_decisionlevelatt':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_DecisionLevelAtt', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_emb32':
        iteration = 560000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_emb128':
        iteration = 560000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_emb512':
        iteration = 440000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_emb512', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_hop500':
        iteration = 440000
        _load_metrics('main', 32000, 1024, 
            500, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_hop640':
        iteration = 440000
        _load_metrics('main', 32000, 1024, 
            640, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'cnn13_hop1000':
        iteration = 540000
        _load_metrics('main', 32000, 1024, 
            1000, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'mobilenetv1':
        iteration = 560000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'mobilenetv2':
        iteration = 560000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'MobileNetV2', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'resnet18':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet18', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'resnet34':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet34', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'resnet50':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'ResNet50', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'dainet':
        iteration = 600000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_DaiNet', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'leenet':
        iteration = 540000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_LeeNet', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'leenet18':
        iteration = 440000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_LeeNet18', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'resnet34_1d':
        iteration = 500000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet34', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'resnet50_1d':
        iteration = 500000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn1d_ResNet50', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'waveform_cnn2d':
        iteration = 660000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_WavCnn2d', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    elif select == 'waveform_spandwav':
        iteration = 700000
        _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32, iteration)

 
def crop_label(label):
    max_len = 16
    if len(label) <= max_len:
        return label
    else:
        words = label.split(' ')
        cropped_label = ''
        for w in words:
            if len(cropped_label + ' ' + w) > max_len:
                break
            else:
                cropped_label += ' {}'.format(w)
    return cropped_label

def add_comma(integer):
    integer = int(integer)
    if integer >= 1000:
        return str(integer // 1000) + ',' + str(integer % 1000)
    else:
        return str(integer)


def plot_class_iteration(args):
    
    # Arguments & parameters
    workspace = args.workspace
    select = args.select
    
    save_out_path = 'results_map/class_iteration_map.pdf'
    create_folder(os.path.dirname(save_out_path))

    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size, iteration):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))
        return statistics_dict

    iteration = 600000
    statistics_dict = _load_metrics('main', 32000, 1024, 
        320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    mAP_mat = np.array([e['average_precision'] for e in statistics_dict['test']])
    mAP_mat = mAP_mat[0 : 300, :]
    sorted_indexes = np.argsort(config.full_samples_per_class)[::-1]


    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    ranges = [np.arange(0, 10), np.arange(250, 260), np.arange(517, 527)]
    axs[0].set_ylabel('AP')

    for col in range(0, 3):
        axs[col].set_ylim(0, 1.)
        axs[col].set_xlim(0, 301)
        axs[col].set_xlabel('Iterations')
        axs[col].set_ylabel('AP')
        axs[col].xaxis.set_ticks(np.arange(0, 301, 100))
        axs[col].xaxis.set_ticklabels(['0', '200k', '400k', '600k'])
        lines = []
        for _ix in ranges[col]:
            _label = crop_label(config.labels[sorted_indexes[_ix]]) + \
                ' ({})'.format(add_comma(config.full_samples_per_class[sorted_indexes[_ix]]))
            line, = axs[col].plot(mAP_mat[:, sorted_indexes[_ix]], label=_label)
            lines.append(line)
        box = axs[col].get_position()
        axs[col].set_position([box.x0, box.y0, box.width * 1., box.height])
        axs[col].legend(handles=lines, bbox_to_anchor=(1., 1.))
        axs[col].yaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
 
    plt.tight_layout(pad=4, w_pad=1, h_pad=1)
    plt.savefig(save_out_path)
    print(save_out_path)


def _load_old_metrics(workspace, filename, iteration, data_type):
    
    assert data_type in ['train', 'test']
    
    stat_name = "stat_{}_iters.p".format(iteration)

    # Load stats
    stat_path = os.path.join(workspace, "stats", filename, data_type, stat_name)
    try:
        stats = cPickle.load(open(stat_path, 'rb'))
    except:
        stats = cPickle.load(open(stat_path, 'rb'), encoding='latin1')

    precisions = [stat['precisions'] for stat in stats]
    recalls = [stat['recalls'] for stat in stats]
    maps = np.array([stat['AP'] for stat in stats])
    aucs = np.array([stat['auc'] for stat in stats])
    
    return {'average_precision': maps, 'AUC': aucs}

def _sort(ys):
    sorted_idxes = np.argsort(ys)
    sorted_idxes = sorted_idxes[::-1]
    sorted_ys = ys[sorted_idxes]
    sorted_lbs = [config.labels[e] for e in sorted_idxes]
    return sorted_ys, sorted_idxes, sorted_lbs

def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['x'][:]
        y = hf['y'][:]
        video_id_list = list(hf['video_id_list'][:])
    return x, y, video_id_list

def get_avg_stats(workspace, bgn_iter, fin_iter, interval_iter, filename, data_type):
    
    assert data_type in ['train', 'test']
    bal_train_hdf5 = "/vol/vssp/msos/audioset/packed_features/bal_train.h5"
    eval_hdf5 = "/vol/vssp/msos/audioset/packed_features/eval.h5"
    unbal_train_hdf5 = "/vol/vssp/msos/audioset/packed_features/unbal_train.h5"
    
    t1 = time.time()
    if data_type == 'test':
        (te_x, te_y, te_id_list) = load_data(eval_hdf5)
    elif data_type == 'train':
        (te_x, te_y, te_id_list) = load_data(bal_train_hdf5)
    y = te_y
    
    prob_dir = os.path.join(workspace, "probs", filename, data_type)
    names = os.listdir(prob_dir)
    
    probs = []
    iters = range(bgn_iter, fin_iter, interval_iter)
    for iter in iters:
        pickle_path = os.path.join(prob_dir, "prob_%d_iters.p" % iter)
        try:
            prob = cPickle.load(open(pickle_path, 'rb'))
        except:
            prob = cPickle.load(open(pickle_path, 'rb'), encoding='latin1')
        probs.append(prob)
    
    avg_prob = np.mean(np.array(probs), axis=0)
    
    n_out = y.shape[1]
    stats = []
    for k in range(n_out): # around 7 seconds
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y[:, k], avg_prob[:, k])
        avg_precision = metrics.average_precision_score(y[:, k], avg_prob[:, k], average=None)
        (fpr, tpr, thresholds) = metrics.roc_curve(y[:, k], avg_prob[:, k])
        auc = metrics.roc_auc_score(y[:, k], avg_prob[:, k], average=None)
        # eer = pp_data.eer(avg_prob[:, k], y[:, k])
        
        skip = 1000
        dict = {'precisions': precisions[0::skip], 'recalls': recalls[0::skip], 'AP': avg_precision, 
                'fpr': fpr[0::skip], 'fnr': 1. - tpr[0::skip], 'auc': auc}
        
        stats.append(dict)
        
    mAPs = np.array([e['AP'] for e in stats])
    aucs = np.array([e['auc'] for e in stats])
        
    print("Get avg time: {}".format(time.time() - t1))
        
    return {'average_precision': mAPs, 'auc': aucs}


def _samples_num_per_class():
    bal_train_hdf5 = "/vol/vssp/msos/audioset/packed_features/bal_train.h5"
    eval_hdf5 = "/vol/vssp/msos/audioset/packed_features/eval.h5"
    unbal_train_hdf5 = "/vol/vssp/msos/audioset/packed_features/unbal_train.h5"

    (x, y, id_list) = load_data(eval_hdf5)
    eval_num = np.sum(y, axis=0)

    (x, y, id_list) = load_data(bal_train_hdf5)
    bal_num = np.sum(y, axis=0)

    (x, y, id_list) = load_data(unbal_train_hdf5)
    unbal_num = np.sum(y, axis=0)

    return bal_num, unbal_num, eval_num


def get_label_quality():
    
    rate_csv = '/vol/vssp/msos/qk/workspaces/pub_audioset_tagging_cnn_transfer/metadata/qa_true_counts.csv'
    
    with open(rate_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lis = list(reader)
        
    rates = []

    for n in range(1, len(lis)):
        li = lis[n]
        if float(li[1]) == 0:
            rate = None
        else:
            rate = float(li[2]) / float(li[1])
        rates.append(rate)
    
    return rates


def summary_stats(args):
    # Arguments & parameters
    workspace = args.workspace

    out_stat_path = os.path.join(workspace, 'results', 'stats_for_paper.pkl')
    create_folder(os.path.dirname(out_stat_path))

    # Old workspace
    old_workspace = '/vol/vssp/msos/qk/workspaces/audioset_classification'

    # bal_train_metrics = _load_old_metrics(old_workspace, 'tmp127', 20000, 'train')
    # eval_metrics = _load_old_metrics(old_workspace, 'tmp127', 20000, 'test')
    
    bal_train_metrics = get_avg_stats(old_workspace, bgn_iter=10000, fin_iter=50001, interval_iter=5000, filename='tmp127_re', data_type='train')
    eval_metrics = get_avg_stats(old_workspace, bgn_iter=10000, fin_iter=50001, interval_iter=5000, filename='tmp127_re', data_type='test')

    maps0te = eval_metrics['average_precision']
    (maps0te, sorted_idxes, sorted_lbs) = _sort(maps0te)

    bal_num, unbal_num, eval_num = _samples_num_per_class()

    output_dict = {
        'labels': config.labels, 
        'label_quality': get_label_quality(), 
        'sorted_indexes_for_plot': sorted_idxes, 
        'official_balanced_trainig_samples': bal_num, 
        'official_unbalanced_training_samples': unbal_num, 
        'official_eval_samples': eval_num, 
        'downloaded_full_training_samples': config.full_samples_per_class, 
        'averaging_instance_system_avg_9_probs_from_10000_to_50000_iterations': 
            {'bal_train': bal_train_metrics, 'eval': eval_metrics}
        }

    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size, iteration):
        _workspace = '/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer'
        statistics_path = os.path.join(_workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))

        _idx = iteration // 2000
        _dict = {'bal_train': {'average_precision': statistics_dict['bal'][_idx]['average_precision'], 
                                'auc': statistics_dict['bal'][_idx]['auc']}, 
                'eval': {'average_precision': statistics_dict['test'][_idx]['average_precision'], 
                        'auc': statistics_dict['test'][_idx]['auc']}}
        return _dict

    iteration = 600000
    output_dict['cnn13_system_iteration60k'] = _load_metrics('main', 32000, 1024, 
        320, 64, 50, 14000, 'full_train', 'Cnn13', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    iteration = 560000
    output_dict['mobilenetv1_system_iteration56k'] = _load_metrics('main', 32000, 1024, 
        320, 64, 50, 14000, 'full_train', 'MobileNetV1', 'clip_bce', 'balanced', 'mixup', 32, iteration)

    cPickle.dump(output_dict, open(out_stat_path, 'wb'))
    print('Write stats for paper to {}'.format(out_stat_path))

 
def prepare_plot_long_4_rows(sorted_lbs):
    N = len(sorted_lbs)

    f,(ax1a, ax2a, ax3a, ax4a) = plt.subplots(4, 1,sharey=False, facecolor='w', figsize=(10, 12))

    fontsize = 5

    K = 132
    ax1a.set_xlim(0, K)
    ax2a.set_xlim(K, 2 * K)
    ax3a.set_xlim(2 * K, 3 * K)
    ax4a.set_xlim(3 * K, N)
    
    truncated_sorted_lbs = []
    for lb in sorted_lbs:
        lb = lb[0 : 25]
        words = lb.split(' ')
        if len(words[-1]) < 3:
            lb = ' '.join(words[0:-1])
        truncated_sorted_lbs.append(lb)
  
    ax1a.grid(which='major', axis='x', linestyle='-', alpha=0.3)
    ax2a.grid(which='major', axis='x', linestyle='-', alpha=0.3)
    ax3a.grid(which='major', axis='x', linestyle='-', alpha=0.3)
    ax4a.grid(which='major', axis='x', linestyle='-', alpha=0.3)
    
    ax1a.set_yscale('log')
    ax2a.set_yscale('log')
    ax3a.set_yscale('log')
    ax4a.set_yscale('log')
    
    ax1b = ax1a.twinx()
    ax2b = ax2a.twinx()
    ax3b = ax3a.twinx()
    ax4b = ax4a.twinx()
    ax1b.set_ylim(0., 1.)
    ax2b.set_ylim(0., 1.)
    ax3b.set_ylim(0., 1.)
    ax4b.set_ylim(0., 1.)
    ax1b.set_ylabel('Average precision')
    ax2b.set_ylabel('Average precision')
    ax3b.set_ylabel('Average precision')
    ax4b.set_ylabel('Average precision')
    
    ax1b.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    ax2b.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    ax3b.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    ax4b.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    
    ax1a.xaxis.set_ticks(np.arange(K))
    ax1a.xaxis.set_ticklabels(truncated_sorted_lbs[0:K], rotation=90, fontsize=fontsize)
    ax1a.xaxis.tick_bottom()
    ax1a.set_ylabel("Number of audio clips")
    
    ax2a.xaxis.set_ticks(np.arange(K, 2*K))
    ax2a.xaxis.set_ticklabels(truncated_sorted_lbs[K:2*K], rotation=90, fontsize=fontsize)
    ax2a.xaxis.tick_bottom()
    # ax2a.tick_params(left='off', which='both')
    ax2a.set_ylabel("Number of audio clips")
    
    ax3a.xaxis.set_ticks(np.arange(2*K, 3*K))
    ax3a.xaxis.set_ticklabels(truncated_sorted_lbs[2*K:3*K], rotation=90, fontsize=fontsize)
    ax3a.xaxis.tick_bottom()
    ax3a.set_ylabel("Number of audio clips")
    
    ax4a.xaxis.set_ticks(np.arange(3*K, N))
    ax4a.xaxis.set_ticklabels(truncated_sorted_lbs[3*K:], rotation=90, fontsize=fontsize)
    ax4a.xaxis.tick_bottom()
    # ax4a.tick_params(left='off', which='both')
    ax4a.set_ylabel("Number of audio clips")
    
    ax1a.spines['right'].set_visible(False)
    ax1b.spines['right'].set_visible(False)
    ax2a.spines['left'].set_visible(False)
    ax2b.spines['left'].set_visible(False)
    ax2a.spines['right'].set_visible(False)
    ax2b.spines['right'].set_visible(False)
    ax3a.spines['left'].set_visible(False)
    ax3b.spines['left'].set_visible(False)
    ax3a.spines['right'].set_visible(False)
    ax3b.spines['right'].set_visible(False)
    ax4a.spines['left'].set_visible(False)
    ax4b.spines['left'].set_visible(False)
    
    plt.subplots_adjust(hspace = 0.8)
    
    return ax1a, ax2a, ax3a, ax4a, ax1b, ax2b, ax3b, ax4b

def _scatter_4_rows(x, ax, ax2, ax3, ax4, s, c, marker='.', alpha=1.):
    N = len(x)
    ax.scatter(np.arange(N), x, s=s, c=c, marker=marker, alpha=alpha)
    ax2.scatter(np.arange(N), x, s=s, c=c, marker=marker, alpha=alpha)
    ax3.scatter(np.arange(N), x, s=s, c=c, marker=marker, alpha=alpha)
    ax4.scatter(np.arange(N), x, s=s, c=c, marker=marker, alpha=alpha)

def _plot_4_rows(x, ax, ax2, ax3, ax4, c, linewidth=1.0, alpha=1.0, label=""):
    N = len(x)
    ax.plot(x, c=c, linewidth=linewidth, alpha=alpha)
    ax2.plot(x, c=c, linewidth=linewidth, alpha=alpha)
    ax3.plot(x, c=c, linewidth=linewidth, alpha=alpha)
    line, = ax4.plot(x, c=c, linewidth=linewidth, alpha=alpha, label=label)
    return line

def plot_long_fig(args):
    # Arguments & parameters
    workspace = args.workspace
    
    # Paths
    stat_path = os.path.join(workspace, 'results', 'stats_for_paper.pkl')
    save_out_path = 'results/long_fig.pdf'
    create_folder(os.path.dirname(save_out_path))

    # Stats
    stats = cPickle.load(open(stat_path, 'rb'))

    N = len(config.labels)
    sorted_indexes = stats['sorted_indexes_for_plot']
    sorted_labels = np.array(config.labels)[sorted_indexes]
    audio_clips_per_class = stats['official_balanced_trainig_samples'] + stats['official_unbalanced_training_samples']
    audio_clips_per_class = audio_clips_per_class[sorted_indexes]

    (ax1a, ax2a, ax3a, ax4a, ax1b, ax2b, ax3b, ax4b) = prepare_plot_long_4_rows(sorted_labels)
 
    # plot the same data on both axes
    ax1a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
    ax2a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
    ax3a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
    ax4a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
   
    maps_avg_instances = stats['averaging_instance_system_avg_9_probs_from_10000_to_50000_iterations']['eval']['average_precision']
    maps_avg_instances = maps_avg_instances[sorted_indexes]

    maps_cnn13 = stats['cnn13_system_iteration60k']['eval']['average_precision']
    maps_cnn13 = maps_cnn13[sorted_indexes]

    maps_mobilenetv1 = stats['mobilenetv1_system_iteration56k']['eval']['average_precision']
    maps_mobilenetv1 = maps_mobilenetv1[sorted_indexes]

    maps_logmel_wavegram_cnn = _load_metrics0_classwise('main', 32000, 1024, 
        320, 64, 50, 14000, 'full_train', 'Cnn13_SpAndWav', 'clip_bce', 'balanced', 'mixup', 32)
    maps_logmel_wavegram_cnn = maps_logmel_wavegram_cnn[sorted_indexes]

    _scatter_4_rows(maps_avg_instances, ax1b, ax2b, ax3b, ax4b, s=5, c='k')
    _scatter_4_rows(maps_cnn13, ax1b, ax2b, ax3b, ax4b, s=5, c='r')
    _scatter_4_rows(maps_mobilenetv1, ax1b, ax2b, ax3b, ax4b, s=5, c='b')
    _scatter_4_rows(maps_logmel_wavegram_cnn, ax1b, ax2b, ax3b, ax4b, s=5, c='g')
    
    linewidth = 0.7
    line0te = _plot_4_rows(maps_avg_instances, ax1b, ax2b, ax3b, ax4b, c='k', linewidth=linewidth, label='AP with averaging instances (baseline)')
    line1te = _plot_4_rows(maps_cnn13, ax1b, ax2b, ax3b, ax4b, c='r', linewidth=linewidth, label='AP with CNN14')
    line2te = _plot_4_rows(maps_mobilenetv1, ax1b, ax2b, ax3b, ax4b, c='b', linewidth=linewidth, label='AP with MobileNetV1')
    line3te = _plot_4_rows(maps_logmel_wavegram_cnn, ax1b, ax2b, ax3b, ax4b, c='g', linewidth=linewidth, label='AP with Wavegram-Logmel-CNN')

    label_quality = stats['label_quality']
    sorted_rate = np.array(label_quality)[sorted_indexes]
    for k in range(len(sorted_rate)):
        if sorted_rate[k] and sorted_rate[k] == 1:
            sorted_rate[k] = 0.99
    
    ax1b.scatter(np.arange(N)[sorted_rate != None], sorted_rate[sorted_rate != None], s=12, c='r', linewidth=0.8, marker='+')
    ax2b.scatter(np.arange(N)[sorted_rate != None], sorted_rate[sorted_rate != None], s=12, c='r', linewidth=0.8, marker='+')
    ax3b.scatter(np.arange(N)[sorted_rate != None], sorted_rate[sorted_rate != None], s=12, c='r', linewidth=0.8, marker='+')
    line_label_quality = ax4b.scatter(np.arange(N)[sorted_rate != None], sorted_rate[sorted_rate != None], s=12, c='r', linewidth=0.8, marker='+', label='Label quality')
    ax1b.scatter(np.arange(N)[sorted_rate == None], 0.5 * np.ones(len(np.arange(N)[sorted_rate == None])), s=12, c='r', linewidth=0.8, marker='_')
    ax2b.scatter(np.arange(N)[sorted_rate == None], 0.5 * np.ones(len(np.arange(N)[sorted_rate == None])), s=12, c='r', linewidth=0.8, marker='_')
    ax3b.scatter(np.arange(N)[sorted_rate == None], 0.5 * np.ones(len(np.arange(N)[sorted_rate == None])), s=12, c='r', linewidth=0.8, marker='_')
    ax4b.scatter(np.arange(N)[sorted_rate == None], 0.5 * np.ones(len(np.arange(N)[sorted_rate == None])), s=12, c='r', linewidth=0.8, marker='_')
    
    plt.legend(handles=[line0te, line1te, line2te, line3te, line_label_quality], fontsize=6, loc=1)
    
    plt.savefig(save_out_path)
    print('Save fig to {}'.format(save_out_path))
 
def plot_flops(args):

    # Arguments & parameters
    workspace = args.workspace
    
    # Paths
    save_out_path = 'results_map/flops.pdf'
    create_folder(os.path.dirname(save_out_path))

    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(1, 1)

    model_types = np.array(['Cnn6', 'Cnn10', 'Cnn14', 'ResNet22', 'ResNet38', 'ResNet54', 
        'MobileNetV1', 'MobileNetV2', 'DaiNet', 'LeeNet', 'LeeNet18', 
        'Res1dNet30', 'Res1dNet44', 'Wavegram-CNN', 'Wavegram-\nLogmel-CNN'])
    flops = np.array([21.986, 21.986, 42.220, 30.081, 48.962, 54.563, 3.614, 2.810, 
        30.395, 4.741, 26.369, 32.688, 61.833, 44.234, 53.510])
    mAPs = np.array([0.343, 0.380, 0.431, 0.430, 0.434, 0.429, 0.389, 0.383, 0.295, 
        0.266, 0.336, 0.365, 0.355, 0.389, 0.439])

    sorted_indexes = np.sort(flops)
    ax.scatter(flops, mAPs)

    shift = [[1, 0.002], [1, -0.006], [-1, -0.014], [-2, 0.006], [-7, 0.006], 
        [1, -0.01], [0.5, 0.004], [-1, -0.014], [1, -0.007], [0.8, -0.008], 
        [1, -0.007], [1, 0.002], [-6, -0.015], [1, -0.008], [0.8, 0]]

    for i, model_type in enumerate(model_types):
        ax.annotate(model_type, (flops[i] + shift[i][0], mAPs[i] + shift[i][1]))

    ax.plot(flops[[0, 1, 2]], mAPs[[0, 1, 2]])
    ax.plot(flops[[3, 4, 5]], mAPs[[3, 4, 5]])
    ax.plot(flops[[6, 7]], mAPs[[6, 7]])
    ax.plot(flops[[9, 10]], mAPs[[9, 10]])
    ax.plot(flops[[11, 12]], mAPs[[11, 12]])
    ax.plot(flops[[13, 14]], mAPs[[13, 14]])

    ax.set_xlim(0, 70)
    ax.set_ylim(0.2, 0.5)
    ax.set_xlabel('Multi-adds (million)')
    ax.set_ylabel('mAP')

    plt.tight_layout(0, 0, 0)

    plt.savefig(save_out_path)
    print('Write out figure to {}'.format(save_out_path))


def spearman(args):

    # Arguments & parameters
    workspace = args.workspace

    # Paths
    stat_path = os.path.join(workspace, 'results', 'stats_for_paper.pkl')

    # Stats
    stats = cPickle.load(open(stat_path, 'rb'))

    label_quality = np.array([qu if qu else 0.5 for qu in stats['label_quality']])
    training_samples = np.array(stats['official_balanced_trainig_samples']) + \
        np.array(stats['official_unbalanced_training_samples'])
    mAP = stats['averaging_instance_system_avg_9_probs_from_10000_to_50000_iterations']['eval']['average_precision']

    import scipy
    samples_spearman = scipy.stats.spearmanr(training_samples, mAP)[0]
    quality_spearman = scipy.stats.spearmanr(label_quality, mAP)[0]

    print('Training samples spearman: {:.3f}'.format(samples_spearman))
    print('Quality spearman: {:.3f}'.format(quality_spearman))


def print_results(args):

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn14_mixup_time_domain', 'clip_bce', 'balanced', 'mixup', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'none', 'none', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'none', 'none', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'none', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'balanced_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)

    # 
    (mAP, mAUC, dprime) = _load_metrics0_classwise2('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn13_emb32', 'clip_bce', 'balanced', 'mixup', 32)

    (mAP, mAUC, dprime) = _load_metrics0_classwise2('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn13_emb128', 'clip_bce', 'balanced', 'mixup', 32)

    # partial
    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'partial_0.8_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'partial_0.5_full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)

    # Sample rate
    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn14_16k', 'clip_bce', 'balanced', 'mixup', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 64, 50, 14000, 'full_train', 'Cnn14_8k', 'clip_bce', 'balanced', 'mixup', 32)

    # Mel bins
    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 128, 50, 14000, 'full_train', 'Cnn14_mel128', 'clip_bce', 'balanced', 'mixup', 32)

    (mAP, mAUC, dprime) = _load_metrics_classwise('main', 32000, 1024, 320, 32, 50, 14000, 'full_train', 'Cnn14_mel32', 'clip_bce', 'balanced', 'mixup', 32)

    import crash
    asdf

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--dataset_dir', type=str, required=True)
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--select', type=str, required=True)
    
    parser_plot = subparsers.add_parser('plot_for_paper')
    parser_plot.add_argument('--dataset_dir', type=str, required=True)
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--select', type=str, required=True)

    parser_plot = subparsers.add_parser('plot_for_paper2')
    parser_plot.add_argument('--dataset_dir', type=str, required=True)
    parser_plot.add_argument('--workspace', type=str, required=True)

    parser_values = subparsers.add_parser('plot_class_iteration')
    parser_values.add_argument('--workspace', type=str, required=True)
    parser_values.add_argument('--select', type=str, required=True)

    parser_summary_stats = subparsers.add_parser('summary_stats')
    parser_summary_stats.add_argument('--workspace', type=str, required=True)

    parser_plot_long = subparsers.add_parser('plot_long_fig')
    parser_plot_long.add_argument('--workspace', type=str, required=True)

    parser_plot_flops = subparsers.add_parser('plot_flops')
    parser_plot_flops.add_argument('--workspace', type=str, required=True)
 
    parser_spearman = subparsers.add_parser('spearman')
    parser_spearman.add_argument('--workspace', type=str, required=True)

    parser_print = subparsers.add_parser('print')
    parser_print.add_argument('--workspace', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'plot':
        plot(args)

    elif args.mode == 'plot_for_paper':
        plot_for_paper(args)

    elif args.mode == 'plot_for_paper2':
        plot_for_paper2(args)
        
    elif args.mode == 'table_values':
        table_values(args)

    elif args.mode == 'plot_class_iteration':
        plot_class_iteration(args)

    elif args.mode == 'summary_stats':
        summary_stats(args)

    elif args.mode == 'plot_long_fig':
        plot_long_fig(args)

    elif args.mode == 'plot_flops':
        plot_flops(args)

    elif args.mode == 'spearman':
        spearman(args)

    elif args.mode == 'print':
        print_results(args)

    else:
        raise Exception('Error argument!')