class BaseTxtProcessor:
    @staticmethod
    def sp_phonemes():
        return ['|']

    @classmethod
    def process(cls, txt, pre_align_args):
        raise NotImplementedError
