import re
import jieba
from pypinyin import pinyin, Style
from text_to_speech.utils.text.text_norm import NSWNormalizer
from text_to_speech.data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor, register_txt_processors
from text_to_speech.utils.text.text_encoder import PUNCS, is_sil_phoneme

ALL_SHENMU = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
              'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']


@register_txt_processors('zh')
class TxtProcessor(BaseTxtProcessor):
    table = {ord(f): ord(t) for f, t in zip(
        u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
        u':,.!?[]()%#@&1234567890')}

    @staticmethod
    def sp_phonemes():
        return ['|', '#']

    @staticmethod
    def preprocess_text(text):
        text = text.translate(TxtProcessor.table)
        text = NSWNormalizer(text).normalize(remove_punc=False).lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        text = re.sub(rf"[A-Za-z]+", r"$", text)
        return text

    @classmethod
    def pinyin_with_en(cls, txt, style):
        x = pinyin(txt, style)
        x = [t[0] for t in x]
        x_ = []
        for t in x:
            if '$' not in t:
                x_.append(t)
            else:
                x_ += list(t)
        x_ = [t if t != '$' else 'ENG' for t in x_]
        return x_

    @classmethod
    def process(cls, txt, pre_align_args):
        txt = cls.preprocess_text(txt)
        txt = txt.replace("嗯", "蒽") # pypin会把嗯的声母韵母识别为''，导致ph2word出现错位。
        # https://blog.csdn.net/zhoulei124/article/details/89055403
        
        pre_align_args['use_tone'] = False
        
        shengmu = cls.pinyin_with_en(txt, style=Style.INITIALS)
        yunmu = cls.pinyin_with_en(txt, style=
        Style.FINALS_TONE3 if pre_align_args['use_tone'] else Style.FINALS)
        assert len(shengmu) == len(yunmu)
        for i in range(len(shengmu)):
            if shengmu[i] == '' and yunmu[i] == '':
                print(f"发现了一个声母韵母都是空的文字:{txt[i]}")
        ph_list = []
        for a, b in zip(shengmu, yunmu):

            if b == 'ueng': # 发现sing数据集里没有后鼻音
                b = 'uen'
                
            if a == b:
                ph_list += [a]
            else:
                ph_list += [a + "%" + b]
        seg_list = '#'.join(jieba.cut(txt))
        assert len(ph_list) == len([s for s in seg_list if s != '#']), (ph_list, seg_list)

        # 加入词边界'#'
        ph_list_ = []
        seg_idx = 0
        for p in ph_list:
            if seg_list[seg_idx] == '#':
                ph_list_.append('#')
                seg_idx += 1
            elif len(ph_list_) > 0:
                ph_list_.append("|")
            seg_idx += 1
            finished = False
            if not finished:
                ph_list_ += [x for x in p.split("%") if x != '']

        ph_list = ph_list_

        # 去除静音符号周围的词边界标记 [..., '#', ',', '#', ...]
        sil_phonemes = list(PUNCS) + TxtProcessor.sp_phonemes()
        ph_list_ = []
        for i in range(0, len(ph_list), 1):
            if ph_list[i] != '#' or (ph_list[i - 1] not in sil_phonemes and ph_list[i + 1] not in sil_phonemes):
                ph_list_.append(ph_list[i])
        ph_list = ph_list_

        txt_struct = [[w, []] for w in txt]
        i = 0
        for ph in ph_list:
            if ph == '|' or ph == '#':
                i += 1
                continue
            # elif ph in [',', '.']:
            elif ph in [',', '.', '?', '!', ':']:
                i += 1
                txt_struct[i][1].append(ph)
                i += 1
                continue
            txt_struct[i][1].append(ph)
        # return ph_list, txt
        txt_struct.insert(0, ['_NONE', ['_NONE']])
        txt_struct.append(['breathe', ['breathe']])

        # txt_struct.insert(0, ['<BOS>', ['<BOS>']])
        # txt_struct.append(['<EOS>', ['<EOS>']])
        return txt_struct, txt


if __name__ == '__main__':
    # t = 'simon演唱过后，simon还进行了simon精彩的文艺演出simon.'
    t = '你当我傻啊？脑子那么大怎么塞进去？?？'
    phs, txt = TxtProcessor.process(t, {'use_tone': True})
    print(phs, txt)
