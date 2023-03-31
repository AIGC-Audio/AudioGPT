import os

os.environ["OMP_NUM_THREADS"] = "1"

from data_gen.tts.txt_processors.zh_g2pM import ALL_SHENMU
from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.data_gen_utils import get_mel2ph
from utils.hparams import set_hparams, hparams
import numpy as np


class ZhBinarizer(BaseBinarizer):
    @staticmethod
    def get_align(tg_fn, ph, mel, phone_encoded, res):
        if tg_fn is not None and os.path.exists(tg_fn):
            _, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        ph_list = ph.split(" ")
        assert len(dur) == len(ph_list)
        mel2ph = []
        # 分隔符的时长分配给韵母
        dur_cumsum = np.pad(np.cumsum(dur), [1, 0], mode='constant', constant_values=0)
        for i in range(len(dur)):
            p = ph_list[i]
            if p[0] != '<' and not p[0].isalpha():
                uv_ = res['f0'][dur_cumsum[i]:dur_cumsum[i + 1]] == 0
                j = 0
                while j < len(uv_) and not uv_[j]:
                    j += 1
                dur[i - 1] += j
                dur[i] -= j
                if dur[i] < 100:
                    dur[i - 1] += dur[i]
                    dur[i] = 0
        # 声母和韵母等长
        for i in range(len(dur)):
            p = ph_list[i]
            if p in ALL_SHENMU:
                p_next = ph_list[i + 1]
                if not (dur[i] > 0 and p_next[0].isalpha() and p_next not in ALL_SHENMU):
                    print(f"assert dur[i] > 0 and p_next[0].isalpha() and p_next not in ALL_SHENMU, "
                          f"dur[i]: {dur[i]}, p: {p}, p_next: {p_next}.")
                    continue
                total = dur[i + 1] + dur[i]
                dur[i] = total // 2
                dur[i + 1] = total - dur[i]
        for i in range(len(dur)):
            mel2ph += [i + 1] * dur[i]
        mel2ph = np.array(mel2ph)
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(f"| Align does not match: {(mel2ph.max() - 1, len(phone_encoded))}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur


if __name__ == "__main__":
    set_hparams()
    ZhBinarizer().process()
