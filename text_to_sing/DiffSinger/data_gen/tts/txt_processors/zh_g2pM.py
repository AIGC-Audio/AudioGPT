import re
import jieba
from pypinyin import pinyin, Style
from data_gen.tts.data_gen_utils import PUNCS
from data_gen.tts.txt_processors import zh
from g2pM import G2pM

ALL_SHENMU = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
              'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian',
             'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'ng', 'o', 'ong', 'ou',
             'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've', 'vn']


class TxtProcessor(zh.TxtProcessor):
    model = G2pM()

    @staticmethod
    def sp_phonemes():
        return ['|', '#']

    @classmethod
    def process(cls, txt, pre_align_args):
        txt = cls.preprocess_text(txt)
        ph_list = cls.model(txt, tone=pre_align_args['use_tone'], char_split=True)
        seg_list = '#'.join(jieba.cut(txt))
        assert len(ph_list) == len([s for s in seg_list if s != '#']), (ph_list, seg_list)

        # 加入词边界'#'
        ph_list_ = []
        seg_idx = 0
        for p in ph_list:
            p = p.replace("u:", "v")
            if seg_list[seg_idx] == '#':
                ph_list_.append('#')
                seg_idx += 1
            else:
                ph_list_.append("|")
            seg_idx += 1
            if re.findall('[\u4e00-\u9fff]', p):
                if pre_align_args['use_tone']:
                    p = pinyin(p, style=Style.TONE3, strict=True)[0][0]
                    if p[-1] not in ['1', '2', '3', '4', '5']:
                        p = p + '5'
                else:
                    p = pinyin(p, style=Style.NORMAL, strict=True)[0][0]

            finished = False
            if len([c.isalpha() for c in p]) > 1:
                for shenmu in ALL_SHENMU:
                    if p.startswith(shenmu) and not p.lstrip(shenmu).isnumeric():
                        ph_list_ += [shenmu, p.lstrip(shenmu)]
                        finished = True
                        break
            if not finished:
                ph_list_.append(p)

        ph_list = ph_list_

        # 去除静音符号周围的词边界标记 [..., '#', ',', '#', ...]
        sil_phonemes = list(PUNCS) + TxtProcessor.sp_phonemes()
        ph_list_ = []
        for i in range(0, len(ph_list), 1):
            if ph_list[i] != '#' or (ph_list[i - 1] not in sil_phonemes and ph_list[i + 1] not in sil_phonemes):
                ph_list_.append(ph_list[i])
        ph_list = ph_list_
        return ph_list, txt


if __name__ == '__main__':
    phs, txt = TxtProcessor.process('他来到了，网易杭研大厦', {'use_tone': True})
    print(phs)
