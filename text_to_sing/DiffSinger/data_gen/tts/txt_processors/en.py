import re
from data_gen.tts.data_gen_utils import PUNCS
from g2p_en import G2p
import unicodedata
from g2p_en.expand import normalize_numbers
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor


class EnG2p(G2p):
    word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text):
        # preprocessing
        words = EnG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


class TxtProcessor(BaseTxtProcessor):
    g2p = EnG2p()

    @staticmethod
    def preprocess_text(text):
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ a-z{PUNCS}]", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = text.replace("i.e.", "that is")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        return text

    @classmethod
    def process(cls, txt, pre_align_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls.g2p(txt)
        phs_ = []
        n_word_sep = 0
        for p in phs:
            if p.strip() == '':
                phs_ += ['|']
                n_word_sep += 1
            else:
                phs_ += p.split(" ")
        phs = phs_
        assert n_word_sep + 1 == len(txt.split(" ")), (phs, f"\"{txt}\"")
        return phs, txt
