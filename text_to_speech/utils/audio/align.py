import re

import torch
import numpy as np
from textgrid import TextGrid

from text_to_speech.utils.text.text_encoder import is_sil_phoneme


def get_mel2ph(tg_fn, ph, mel, hop_size, audio_sample_rate, min_sil_duration=0):
    ph_list = ph.split(" ")
    itvs = TextGrid.fromFile(tg_fn)[1]
    itvs_ = []
    for i in range(len(itvs)):
        if itvs[i].maxTime - itvs[i].minTime < min_sil_duration and i > 0 and is_sil_phoneme(itvs[i].mark):
            itvs_[-1].maxTime = itvs[i].maxTime
        else:
            itvs_.append(itvs[i])
    itvs.intervals = itvs_
    itv_marks = [itv.mark for itv in itvs]
    tg_len = len([x for x in itvs if not is_sil_phoneme(x.mark)])
    ph_len = len([x for x in ph_list if not is_sil_phoneme(x)])
    assert tg_len == ph_len, (tg_len, ph_len, itv_marks, ph_list, tg_fn)
    mel2ph = np.zeros([mel.shape[0]], int)

    # to process aishell3_no_tone
    # for _ in range(10):
        
    #     if itvs[-2].mark == '':
    #         start_time = itvs[-2].minTime
    #         end_time = itvs[-1].maxTime
    #         mark = itvs[-1].mark
    #         itvs[-2].maxTime = end_time
    #         itvs[-2].mark = mark
    #         itvs_ = []
    #         for i in range(len(itvs)-1):
    #             itvs_.append(itvs[i])
    #         itvs.intervals = itvs_

    #     if itvs[-1].mark == '':
    #         start_time = itvs[-2].minTime
    #         end_time = itvs[-1].maxTime
    #         itvs[-2].maxTime = end_time
    #         itvs_ = []
    #         for i in range(len(itvs)-1):
    #             itvs_.append(itvs[i])
    #         itvs.intervals = itvs_


    i_itv = 0
    i_ph = 0
    while i_itv < len(itvs):
        itv = itvs[i_itv]
        ph = ph_list[i_ph]
        itv_ph = itv.mark
        start_frame = int(itv.minTime * audio_sample_rate / hop_size + 0.5)
        end_frame = int(itv.maxTime * audio_sample_rate / hop_size + 0.5)
        if is_sil_phoneme(itv_ph) and not is_sil_phoneme(ph):
            mel2ph[start_frame:end_frame] = i_ph
            i_itv += 1
        elif not is_sil_phoneme(itv_ph) and is_sil_phoneme(ph):
            i_ph += 1
        else:
            if not ((is_sil_phoneme(itv_ph) and is_sil_phoneme(ph)) \
                    or re.sub(r'\d+', '', itv_ph.lower()) == re.sub(r'\d+', '', ph.lower())):
                print(f"| WARN: {tg_fn} phs are not same: ", itv_ph, ph, itv_marks, ph_list)
            mel2ph[start_frame:end_frame] = i_ph + 1
            i_ph += 1
            i_itv += 1
    mel2ph[-1] = mel2ph[-2]
    assert not np.any(mel2ph == 0)
    T_t = len(ph_list)
    dur = mel2token_to_dur(mel2ph, T_t)
    return mel2ph.tolist(), dur.tolist()


def split_audio_by_mel2ph(audio, mel2ph, hop_size, audio_num_mel_bins):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if isinstance(mel2ph, torch.Tensor):
        mel2ph = mel2ph.numpy()
    assert len(audio.shape) == 1, len(mel2ph.shape) == 1
    split_locs = []
    for i in range(1, len(mel2ph)):
        if mel2ph[i] != mel2ph[i - 1]:
            split_loc = i * hop_size
            split_locs.append(split_loc)

    new_audio = []
    for i in range(len(split_locs) - 1):
        new_audio.append(audio[split_locs[i]:split_locs[i + 1]])
        new_audio.append(np.zeros([0.5 * audio_num_mel_bins]))
    return np.concatenate(new_audio)


def mel2token_to_dur(mel2token, T_txt=None, max_dur=None):
    is_torch = isinstance(mel2token, torch.Tensor)
    has_batch_dim = True
    if not is_torch:
        mel2token = torch.LongTensor(mel2token)
    if T_txt is None:
        T_txt = mel2token.max()
    if len(mel2token.shape) == 1:
        mel2token = mel2token[None, ...]
        has_batch_dim = False
    B, _ = mel2token.shape
    dur = mel2token.new_zeros(B, T_txt + 1).scatter_add(1, mel2token, torch.ones_like(mel2token))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    if not is_torch:
        dur = dur.numpy()
    if not has_batch_dim:
        dur = dur[0]
    return dur
