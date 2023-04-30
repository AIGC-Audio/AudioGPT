
import random
import torchaudio
from torch._six import string_classes
import collections
import re
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from wav_evaluation.models.utils import read_config_as_args
from wav_evaluation.models.clap import CLAP
import math
import torchaudio.transforms as T
import os
import torch
from importlib_resources import files


class CLAPWrapper():
    """
    A class for interfacing CLAP model.  
    """

    def __init__(self, model_fp,config_path, use_cuda=False):
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        with open(config_path,'r') as f:
            self.config_as_str = f.read()
        self.model_fp = model_fp
        self.use_cuda = use_cuda
        self.clap, self.tokenizer, self.args = self.load_clap()

    def load_clap(self):
        r"""Load CLAP model with args from config file"""

        args = read_config_as_args(self.config_as_str, is_config_str=True)

        if 'bert' in args.text_model:
            self.token_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        else:
            self.token_keys = ['input_ids', 'attention_mask']

        clap = CLAP(
            audioenc_name=args.audioenc_name,
            sample_rate=args.sampling_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            classes_num=args.num_classes,
            out_emb=args.out_emb,
            text_model=args.text_model,
            transformer_embed_dim=args.transformer_embed_dim,
            d_proj=args.d_proj
        )


        # Load pretrained weights for model
        model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']
        clap.load_state_dict(model_state_dict)
        clap.eval()  # set clap in eval mode
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)

        if self.use_cuda and torch.cuda.is_available():
            clap = clap.cuda()

        return clap, tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def resample_and_duration(self,wav_sr,audio_duration,resample=False):
        audio_time_series,sample_rate = wav_sr
        resample_rate = self.args.sampling_rate
        if resample:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        return self.resample_and_duration((audio_time_series, sample_rate),audio_duration,resample)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            if isinstance(audio_file,str):
                audio_tensor = self.load_audio_into_tensor(audio_file, self.args.duration, resample)
            elif isinstance(audio_file,tuple):
                audio_tensor = self.resample_and_duration(audio_file, self.args.duration, resample)
            else: 
                raise TypeError(f"type of audiofile is {type(audio_file)},which is not supported")
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def preprocess_text(self, text_queries):
        r"""Load list of class labels and return tokenized text"""
        tokenized_texts = []
        for ttext in text_queries:
            tok = self.tokenizer.encode_plus(
                text=ttext, add_special_tokens=True, max_length=self.args.text_len, padding="max_length", return_tensors="pt") # max_length=self.args.text_len, padding=True,
            for key in self.token_keys:
                tok[key] = tok[key].reshape(-1).cuda() if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)

    def get_text_embeddings(self, class_labels):
        r"""Load list of class labels and return text embeddings"""
        preprocessed_text = self.preprocess_text(class_labels)
        text_embeddings = self._get_text_embeddings(preprocessed_text)
        text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
        return text_embeddings

    def get_audio_embeddings(self, audio_files, resample):
        r"""Load list of audio files and return a audio embeddings"""
        preprocessed_audio = self.preprocess_audio(audio_files, resample)
        audio_embeddings = self._get_audio_embeddings(preprocessed_audio)
        audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
        return audio_embeddings

    def _get_text_embeddings(self, preprocessed_text):
        r"""Load preprocessed text and return text embeddings"""
        with torch.no_grad():
            text_embeddings = self.clap.caption_encoder(preprocessed_text)
            text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
            return text_embeddings

    def _get_audio_embeddings(self, preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
            #Append [0] the audio emebdding, [1] has output class probabilities
            audio_embeddings = self.clap.audio_encoder(preprocessed_audio)[0]
            audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
            return audio_embeddings
    
    def compute_similarity(self, audio_embeddings, text_embeddings,use_logit_scale = True):
        r"""Compute similarity between text and audio embeddings"""
        if use_logit_scale:
            logit_scale = self.clap.logit_scale.exp()
            similarity = logit_scale*text_embeddings @ audio_embeddings.T
        else:
            similarity = text_embeddings @ audio_embeddings.T
        return similarity.T

    def cal_clap_score(self,txt,audio_path):
        text_embeddings = self.get_text_embeddings([txt])# 经过了norm的embedding
        audio_embeddings = self.get_audio_embeddings([audio_path], resample=True)# 这一步比较耗时，读取音频并重采样到44100
        score = self.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False).squeeze().cpu().numpy()
        return score

    def _generic_batch_inference(self, func, *args):
        r"""Process audio and/or text per batch"""
        input_tmp = args[0]
        batch_size = args[-1]
        # args[0] has audio_files, args[1] has class_labels
        inputs = [args[0], args[1]] if len(args) == 3 else [args[0]]
        args0_len = len(args[0])
        # compute text_embeddings once for all the audio_files batches
        if len(inputs) == 2:
            text_embeddings = self.get_text_embeddings(args[1])
            inputs = [args[0], args[1], text_embeddings]
        dataset_idx = 0
        for _ in range(math.ceil(args0_len/batch_size)):
            next_batch_idx = dataset_idx + batch_size
            # batch size is bigger than available audio/text items
            if next_batch_idx >= args0_len:
                inputs[0] = input_tmp[dataset_idx:]
                return func(*tuple(inputs))
            else:
                inputs[0] = input_tmp[dataset_idx:next_batch_idx]
                yield func(*tuple(inputs))
            dataset_idx = next_batch_idx

    def get_audio_embeddings_per_batch(self, audio_files, batch_size):
        r"""Load preprocessed audio and return a audio embeddings per batch"""
        return self._generic_batch_inference(self.get_audio_embeddings, audio_files, batch_size)

    def get_text_embeddings_per_batch(self, class_labels, batch_size):
        r"""Load preprocessed text and return text embeddings per batch"""
        return self._generic_batch_inference(self.get_text_embeddings, class_labels, batch_size)

    def classify_audio_files_per_batch(self, audio_files, class_labels, batch_size):
        r"""Compute classification probabilities for each audio recording in a batch and each class label"""
        return self._generic_batch_inference(self.classify_audio_files, audio_files, class_labels, batch_size)
