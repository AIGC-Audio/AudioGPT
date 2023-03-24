from data_gen.tts.data_gen_utils import is_sil_phoneme

REGISTERED_TEXT_PROCESSORS = {}

def register_txt_processors(name):
    def _f(cls):
        REGISTERED_TEXT_PROCESSORS[name] = cls
        return cls

    return _f


def get_txt_processor_cls(name):
    return REGISTERED_TEXT_PROCESSORS.get(name, None)


class BaseTxtProcessor:
    @staticmethod
    def sp_phonemes():
        return ['|']

    @classmethod
    def process(cls, txt, preprocess_args):
        raise NotImplementedError

    @classmethod
    def postprocess(cls, txt_struct, preprocess_args):
        # remove sil phoneme in head and tail
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
            txt_struct = txt_struct[:-1]
        if preprocess_args['with_phsep']:
            txt_struct = cls.add_bdr(txt_struct)
        if preprocess_args['add_eos_bos']:
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_