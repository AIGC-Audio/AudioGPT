import utils.commons.single_thread_env  # NOQA
from text_to_speech.utils.commons.hparams import set_hparams, hparams
from text_to_speech.data_gen.tts.runs.binarize import binarize
from text_to_speech.data_gen.tts.runs.preprocess import preprocess
from text_to_speech.data_gen.tts.runs.train_mfa_align import train_mfa_align

if __name__ == '__main__':
    set_hparams()
    preprocess()
    if hparams['preprocess_args']['use_mfa']:
        train_mfa_align()
    binarize()
