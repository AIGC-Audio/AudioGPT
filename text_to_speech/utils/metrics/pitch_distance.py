import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import torch


@jit
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf
    eps = 1e-4
    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1]):
            dtw[i, j] = costs[i, j] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw


def align_from_distances(distance_matrix, debug=False, return_mindist=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0] - 1
    j = distance_matrix.shape[1] - 1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i - 1, j), (i, j - 1), (i - 1, j - 1)], key=lambda x: dtw[x[0], x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)), results] = 1
        plt.matshow(visual)
        plt.show()
    if return_mindist:
        return results, dtw[-1, -1]
    return results


def get_local_context(input_f, max_window=32, scale_factor=1.):
    # input_f: [S, 1], support numpy array or torch tensor
    # return hist: [S, max_window * 2], list of list
    T = input_f.shape[0]
    # max_window = int(max_window * scale_factor)
    derivative = [[0 for _ in range(max_window * 2)] for _ in range(T)]

    for t in range(T):  # travel the time series
        for feat_idx in range(-max_window, max_window):
            if t + feat_idx < 0 or t + feat_idx >= T:
                value = 0
            else:
                value = input_f[t + feat_idx]
            derivative[t][feat_idx + max_window] = value
    return derivative


def cal_localnorm_dist(src, tgt, src_len, tgt_len):
    local_src = torch.tensor(get_local_context(src))
    local_tgt = torch.tensor(get_local_context(tgt, scale_factor=tgt_len / src_len))

    local_norm_src = (local_src - local_src.mean(-1).unsqueeze(-1))  # / local_src.std(-1).unsqueeze(-1)  # [T1, 32]
    local_norm_tgt = (local_tgt - local_tgt.mean(-1).unsqueeze(-1))  # / local_tgt.std(-1).unsqueeze(-1)  # [T2, 32]

    dists = torch.cdist(local_norm_src[None, :, :], local_norm_tgt[None, :, :])  # [1, T1, T2]
    return dists


## here is API for one sample
def LoNDTWDistance(src, tgt):
    # src: [S]
    # tgt: [T]
    dists = cal_localnorm_dist(src, tgt, src.shape[0], tgt.shape[0])  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment, min_distance = align_from_distances(costs.T.cpu().detach().numpy(), return_mindist=True)  # [T]
    return alignment, min_distance

# if __name__ == '__main__':
#     # utils from ns
#     from text_to_speech.utils.pitch_utils import denorm_f0
#     from tasks.singing.fsinging import FastSingingDataset
#     from text_to_speech.utils.hparams import hparams, set_hparams
#
#     set_hparams()
#
#     train_ds = FastSingingDataset('test')
#
#     # Test One sample case
#     sample = train_ds[0]
#     amateur_f0 = sample['f0']
#     prof_f0 = sample['prof_f0']
#
#     amateur_uv = sample['uv']
#     amateur_padding = sample['mel2ph'] == 0
#     prof_uv = sample['prof_uv']
#     prof_padding = sample['prof_mel2ph'] == 0
#     amateur_f0_denorm = denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)
#     prof_f0_denorm = denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)
#     alignment, min_distance = LoNDTWDistance(amateur_f0_denorm, prof_f0_denorm)
#     print(min_distance)
# python utils/pitch_distance.py --config egs/datasets/audio/molar/svc_ppg.yaml
