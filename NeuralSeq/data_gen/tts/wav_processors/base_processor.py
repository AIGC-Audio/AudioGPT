REGISTERED_WAV_PROCESSORS = {}


def register_wav_processors(name):
    def _f(cls):
        REGISTERED_WAV_PROCESSORS[name] = cls
        return cls

    return _f


def get_wav_processor_cls(name):
    return REGISTERED_WAV_PROCESSORS.get(name, None)


class BaseWavProcessor:
    @property
    def name(self):
        raise NotImplementedError

    def output_fn(self, input_fn):
        return f'{input_fn[:-4]}_{self.name}.wav'

    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        raise NotImplementedError
