import utils.commons.single_thread_env  # NOQA
from text_to_speech.utils.commons.hparams import hparams, set_hparams
import importlib


def binarize():
    binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == '__main__':
    set_hparams()
    binarize()
