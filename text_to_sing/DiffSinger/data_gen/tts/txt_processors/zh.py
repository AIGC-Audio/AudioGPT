import re
from pypinyin import pinyin, Style
from data_gen.tts.data_gen_utils import PUNCS
from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor
from utils.text_norm import NSWNormalizer


class TxtProcessor(BaseTxtProcessor):
    table = {ord(f): ord(t) for f, t in zip(
        u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
        u':,.!?[]()%#@&1234567890')}

    @staticmethod
    def preprocess_text(text):
        text = text.translate(TxtProcessor.table)
        text = NSWNormalizer(text).normalize(remove_punc=False)
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        return text

    @classmethod
    def process(cls, txt, pre_align_args):
        txt = cls.preprocess_text(txt)
        shengmu = pinyin(txt, style=Style.INITIALS)  # https://blog.csdn.net/zhoulei124/article/details/89055403
        yunmu_finals = pinyin(txt, style=Style.FINALS)
        yunmu_tone3 = pinyin(txt, style=Style.FINALS_TONE3)
        yunmu = [[t[0] + '5'] if t[0] == f[0] else t for f, t in zip(yunmu_finals, yunmu_tone3)] \
            if pre_align_args['use_tone'] else yunmu_finals

        assert len(shengmu) == len(yunmu)
        phs = ["|"]
        for a, b, c in zip(shengmu, yunmu, yunmu_finals):
            if a[0] == c[0]:
                phs += [a[0], "|"]
            else:
                phs += [a[0], b[0], "|"]
        return phs, txt
