import os
import sys
import numpy as np
import argparse
import h5py
import time
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

from utilities import (create_folder, get_filename, d_prime)
import config


def load_statistics(statistics_path):
    statistics_dict = pickle.load(open(statistics_path, 'rb'))

    bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
    bal_map = np.mean(bal_map, axis=-1)
    test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
    test_map = np.mean(test_map, axis=-1)

    return bal_map, test_map


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
    """E.g., 1234567 -> 1,234,567
    """
    integer = int(integer)
    if integer >= 1000:
        return str(integer // 1000) + ',' + str(integer % 1000)
    else:
        return str(integer)


def plot_classwise_iteration_map(args):
    
    # Paths
    save_out_path = 'results/classwise_iteration_map.pdf'
    create_folder(os.path.dirname(save_out_path))

    # Load statistics
    statistics_dict = pickle.load(open('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_WavegramLogmelCnn_balanced_mixup_bs32.pkl', 'rb'))

    mAP_mat = np.array([e['average_precision'] for e in statistics_dict['test']])
    mAP_mat = mAP_mat[0 : 300, :]   # 300 * 2000 = 600k iterations
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


def plot_six_figures(args):
    
    # Arguments & parameters
    classes_num = config.classes_num
    labels = config.labels
    max_plot_iteration = 540000
    iterations = np.arange(0, max_plot_iteration, 2000)

    # Paths
    class_labels_indices_path = os.path.join('metadata', 'class_labels_indices.csv')
    save_out_path = 'results/six_figures.pdf'
    create_folder(os.path.dirname(save_out_path))
    
    # Plot
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    bal_alpha = 0.3
    test_alpha = 1.0
    linewidth = 1.

    # (a) Comparison of architectures
    if True:
        lines = []

        # Wavegram-Logmel-CNN
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_WavegramLogmelCnn_balanced_mixup_bs32.pkl')
        line, = ax[0, 0].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 0].plot(test_map, label='Wavegram-Logmel-CNN', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Cnn14
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[0, 0].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 0].plot(test_map, label='CNN14', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # MobileNetV1
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_MobileNetV1_balanced_mixup_bs32.pkl')
        line, = ax[0, 0].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 0].plot(test_map, label='MobileNetV1', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0, 0].legend(handles=lines, loc=2)
        ax[0, 0].set_title('(a) Comparison of architectures')

    # (b) Comparison of training data and augmentation'
    if True:
        lines = []

        # Full data + balanced sampler + mixup
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[0, 1].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,mixup (1.9m)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Full data + balanced sampler + mixup in time domain
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_timedomain_bs32.pkl')
        line, = ax[0, 1].plot(bal_map, color='y', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,mixup-wav (1.9m)', color='y', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Full data + balanced sampler + no mixup
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_nomixup_bs32.pkl')
        line, = ax[0, 1].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,no-mixup (1.9m)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Full data + uniform sampler + no mixup
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_nobalanced_nomixup_bs32.pkl')
        line, = ax[0, 1].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,no-bal,no-mixup (1.9m)', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Balanced data + balanced sampler + mixup
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_balanced_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[0, 1].plot(bal_map, color='m', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,mixup (20k)', color='m', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Balanced data + balanced sampler + no mixup
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_balanced_train_Cnn14_balanced_nomixup_bs32.pkl')
        line, = ax[0, 1].plot(bal_map, color='k', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 1].plot(test_map, label='CNN14,bal,no-mixup (20k)', color='k', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0, 1].legend(handles=lines, loc=2, fontsize=8)
        ax[0, 1].set_title('(b) Comparison of training data and augmentation')

    # (c) Comparison of embedding size
    if True:
        lines = []

        # Embedding size 2048
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[0, 2].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 2].plot(test_map, label='CNN14,emb=2048', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Embedding size 128
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_emb128_balanced_mixup_bs32.pkl')
        line, = ax[0, 2].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 2].plot(test_map, label='CNN14,emb=128', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Embedding size 32
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_emb32_balanced_mixup_bs32.pkl')
        line, = ax[0, 2].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0, 2].plot(test_map, label='CNN14,emb=32', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0, 2].legend(handles=lines, loc=2)
        ax[0, 2].set_title('(c) Comparison of embedding size')

    # (d) Comparison of amount of training data
    if True:
        lines = []

        # 100% of full training data
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 0].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 0].plot(test_map, label='CNN14 (100% full)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # 80% of full training data
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_0.8full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 0].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 0].plot(test_map, label='CNN14 (80% full)', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # 50% of full training data
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_0.5full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 0].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 0].plot(test_map, label='cnn14 (50% full)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[1, 0].legend(handles=lines, loc=2)
        ax[1, 0].set_title('(d) Comparison of amount of training data')

    # (e) Comparison of sampling rate
    if True:
        lines = []

        # Cnn14 + 32 kHz
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 1].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 1].plot(test_map, label='CNN14,32kHz', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Cnn14 + 16 kHz
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_16k_balanced_mixup_bs32.pkl')
        line, = ax[1, 1].plot(bal_map, color='b', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 1].plot(test_map, label='CNN14,16kHz', color='b', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Cnn14 + 8 kHz
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_8k_balanced_mixup_bs32.pkl')
        line, = ax[1, 1].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 1].plot(test_map, label='CNN14,8kHz', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[1, 1].legend(handles=lines, loc=2)
        ax[1, 1].set_title('(e) Comparison of sampling rate')

    # (f) Comparison of mel bins number
    if True:
        lines = []

        # Cnn14 + 128 mel bins
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel128_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 2].plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax[1, 2].plot(test_map, label='CNN14,128-melbins', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Cnn14 + 64 mel bins
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel64_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 2].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1, 2].plot(test_map, label='CNN14,64-melbins', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        # Cnn14 + 32 mel bins
        (bal_map, test_map) = load_statistics('paper_statistics/statistics_sr32000_window1024_hop320_mel32_fmin50_fmax14000_full_train_Cnn14_balanced_mixup_bs32.pkl')
        line, = ax[1, 2].plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax[1, 2].plot(test_map, label='CNN14,32-melbins', color='b', alpha=test_alpha, linewidth=linewidth)
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
            ax[i, j].xaxis.set_ticklabels(['0', '100k', '200k', '300k', '400k', '500k'])
            ax[i, j].yaxis.set_ticks(np.arange(0, 0.81, 0.05))
            ax[i, j].yaxis.set_ticklabels(['0', '', '0.1', '', '0.2', '', '0.3', 
                '', '0.4', '', '0.5', '', '0.6', '', '0.7', '', '0.8'])
            ax[i, j].yaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
            ax[i, j].xaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)

    plt.tight_layout(0, 1, 0)
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def plot_complexity_map(args):
    
    # Paths
    save_out_path = 'results/complexity_mAP.pdf'
    create_folder(os.path.dirname(save_out_path))

    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(1, 1)

    model_types = np.array(['Cnn6', 'Cnn10', 'Cnn14', 'ResNet22', 'ResNet38', 'ResNet54', 
        'MobileNetV1', 'MobileNetV2', 'DaiNet', 'LeeNet', 'LeeNet18', 
        'Res1dNet30', 'Res1dNet44', 'Wavegram-CNN', 'Wavegram-\nLogmel-CNN'])
    flops = np.array([21.986, 28.166, 42.220, 30.081, 48.962, 54.563, 3.614, 2.810, 
        30.395, 4.741, 26.369, 32.688, 61.833, 44.234, 53.510])
    mAPs = np.array([0.343, 0.380, 0.431, 0.430, 0.434, 0.429, 0.389, 0.383, 0.295, 
        0.266, 0.336, 0.365, 0.355, 0.389, 0.439])

    sorted_indexes = np.sort(flops)
    ax.scatter(flops, mAPs)

    shift = [[-5.5, -0.004], [1, -0.004], [-1, -0.014], [-2, 0.006], [-7, 0.006], 
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
    ax.set_xlabel('Multi-load_statisticss (million)', fontsize=15)
    ax.set_ylabel('mAP', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout(0, 0, 0)

    plt.savefig(save_out_path)
    print('Write out figure to {}'.format(save_out_path))


def plot_long_fig(args):
    
    # Paths
    stats = pickle.load(open('paper_statistics/stats_for_long_fig.pkl', 'rb'))

    save_out_path = 'results/long_fig.pdf'
    create_folder(os.path.dirname(save_out_path))

    # Load meta
    N = len(config.labels)
    sorted_indexes = stats['sorted_indexes_for_plot']
    sorted_labels = np.array(config.labels)[sorted_indexes]
    audio_clips_per_class = stats['official_balanced_training_samples'] + stats['official_unbalanced_training_samples']
    audio_clips_per_class = audio_clips_per_class[sorted_indexes]

    # Prepare axes for plot
    (ax1a, ax2a, ax3a, ax4a, ax1b, ax2b, ax3b, ax4b) = prepare_plot_long_4_rows(sorted_labels)
 
    # plot the number of training samples
    ax1a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
    ax2a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
    ax3a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
    ax4a.bar(np.arange(N), audio_clips_per_class, alpha=0.3)
   
    # Load mAP of different systems
    """Average instance system of [1] with an mAP of 0.317.
    [1] Kong, Qiuqiang, Changsong Yu, Yong Xu, Turab Iqbal, Wenwu Wang, and 
    Mark D. Plumbley. "Weakly labelled audioset tagging with attention neural 
    networks." IEEE/ACM Transactions on Audio, Speech, and Language Processing 
    27, no. 11 (2019): 1791-1802."""
    maps_avg_instances = stats['averaging_instance_system_avg_9_probs_from_10000_to_50000_iterations']['eval']['average_precision']
    maps_avg_instances = maps_avg_instances[sorted_indexes]

    # PANNs Cnn14
    maps_panns_cnn14 = stats['panns_cnn14']['eval']['average_precision']
    maps_panns_cnn14 = maps_panns_cnn14[sorted_indexes]

    # PANNs MobileNetV1
    maps_panns_mobilenetv1 = stats['panns_mobilenetv1']['eval']['average_precision']
    maps_panns_mobilenetv1 = maps_panns_mobilenetv1[sorted_indexes]

    # PANNs Wavegram-Logmel-Cnn14
    maps_panns_wavegram_logmel_cnn14 = stats['panns_wavegram_logmel_cnn14']['eval']['average_precision']
    maps_panns_wavegram_logmel_cnn14 = maps_panns_wavegram_logmel_cnn14[sorted_indexes]

    # Plot mAPs
    _scatter_4_rows(maps_panns_wavegram_logmel_cnn14, ax1b, ax2b, ax3b, ax4b, s=5, c='g')
    _scatter_4_rows(maps_panns_cnn14, ax1b, ax2b, ax3b, ax4b, s=5, c='r')
    _scatter_4_rows(maps_panns_mobilenetv1, ax1b, ax2b, ax3b, ax4b, s=5, c='b')
    _scatter_4_rows(maps_avg_instances, ax1b, ax2b, ax3b, ax4b, s=5, c='k')
    
    linewidth = 0.7
    line0te = _plot_4_rows(maps_panns_wavegram_logmel_cnn14, ax1b, ax2b, ax3b, ax4b, 
        c='g', linewidth=linewidth, label='AP with Wavegram-Logmel-CNN')
    line1te = _plot_4_rows(maps_panns_cnn14, ax1b, ax2b, ax3b, ax4b, c='r', 
        linewidth=linewidth, label='AP with CNN14')
    line2te = _plot_4_rows(maps_panns_mobilenetv1, ax1b, ax2b, ax3b, ax4b, c='b', 
        linewidth=linewidth, label='AP with MobileNetV1')
    line3te = _plot_4_rows(maps_avg_instances, ax1b, ax2b, ax3b, ax4b, c='k', 
        linewidth=linewidth, label='AP with averaging instances (baseline)')

    # Plot label quality
    label_quality = stats['label_quality']
    sorted_label_quality = np.array(label_quality)[sorted_indexes]
    for k in range(len(sorted_label_quality)):
        if sorted_label_quality[k] and sorted_label_quality[k] == 1:
            sorted_label_quality[k] = 0.99
    
    ax1b.scatter(np.arange(N)[sorted_label_quality != None], 
        sorted_label_quality[sorted_label_quality != None], s=12, c='r', linewidth=0.8, marker='+')
    ax2b.scatter(np.arange(N)[sorted_label_quality != None], 
        sorted_label_quality[sorted_label_quality != None], s=12, c='r', linewidth=0.8, marker='+')
    ax3b.scatter(np.arange(N)[sorted_label_quality != None], 
        sorted_label_quality[sorted_label_quality != None], s=12, c='r', linewidth=0.8, marker='+')
    line_label_quality = ax4b.scatter(np.arange(N)[sorted_label_quality != None], 
        sorted_label_quality[sorted_label_quality != None], s=12, c='r', linewidth=0.8, marker='+', label='Label quality')
    ax1b.scatter(np.arange(N)[sorted_label_quality == None], 
        0.5 * np.ones(len(np.arange(N)[sorted_label_quality == None])), s=12, c='r', linewidth=0.8, marker='_')
    ax2b.scatter(np.arange(N)[sorted_label_quality == None], 
        0.5 * np.ones(len(np.arange(N)[sorted_label_quality == None])), s=12, c='r', linewidth=0.8, marker='_')
    ax3b.scatter(np.arange(N)[sorted_label_quality == None], 
        0.5 * np.ones(len(np.arange(N)[sorted_label_quality == None])), s=12, c='r', linewidth=0.8, marker='_')
    ax4b.scatter(np.arange(N)[sorted_label_quality == None], 
        0.5 * np.ones(len(np.arange(N)[sorted_label_quality == None])), s=12, c='r', linewidth=0.8, marker='_')
    
    plt.legend(handles=[line0te, line1te, line2te, line3te, line_label_quality], fontsize=6, loc=1)
    plt.tight_layout(0, 0, 0)
    plt.savefig(save_out_path)
    print('Save fig to {}'.format(save_out_path))


def prepare_plot_long_4_rows(sorted_lbs):
    N = len(sorted_lbs)

    f,(ax1a, ax2a, ax3a, ax4a) = plt.subplots(4, 1, sharey=False, facecolor='w', figsize=(10, 10.5))

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
    ax2a.set_ylabel("Number of audio clips")
    
    ax3a.xaxis.set_ticks(np.arange(2*K, 3*K))
    ax3a.xaxis.set_ticklabels(truncated_sorted_lbs[2*K:3*K], rotation=90, fontsize=fontsize)
    ax3a.xaxis.tick_bottom()
    ax3a.set_ylabel("Number of audio clips")
    
    ax4a.xaxis.set_ticks(np.arange(3*K, N))
    ax4a.xaxis.set_ticklabels(truncated_sorted_lbs[3*K:], rotation=90, fontsize=fontsize)
    ax4a.xaxis.tick_bottom()
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_classwise_iteration_map = subparsers.add_parser('plot_classwise_iteration_map')
    parser_six_figures = subparsers.add_parser('plot_six_figures')
    parser_complexity_map = subparsers.add_parser('plot_complexity_map')
    parser_long_fig = subparsers.add_parser('plot_long_fig')
    
    args = parser.parse_args()

    if args.mode == 'plot_classwise_iteration_map':
        plot_classwise_iteration_map(args)

    elif args.mode == 'plot_six_figures':
        plot_six_figures(args)
    
    elif args.mode == 'plot_complexity_map':
        plot_complexity_map(args)

    elif args.mode == 'plot_long_fig':
        plot_long_fig(args)

    else:
    	raise Exception('Incorrect argument!')