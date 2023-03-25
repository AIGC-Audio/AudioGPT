import json
import os
import random
from re import L
import traceback
from functools import partial

import numpy as np
from resemblyzer import VoiceEncoder
from tqdm import tqdm

from transformers import AutoTokenizer

# import utils.commons.single_thread_env  # NOQA
from text_to_speech.utils.audio import librosa_wav2spec
from text_to_speech.utils.audio.align import get_mel2ph, mel2token_to_dur
from text_to_speech.utils.audio.cwt import get_lf0_cwt, get_cont_lf0
from text_to_speech.utils.audio.pitch.utils import f0_to_coarse
from text_to_speech.utils.audio.pitch_extractors import extract_pitch_simple
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.commons.indexed_datasets import IndexedDatasetBuilder
from text_to_speech.utils.commons.multiprocess_utils import multiprocess_run_tqdm
from text_to_speech.utils.os_utils import remove_file, copy_file

np.seterr(divide='ignore', invalid='ignore')


class BinarizationError(Exception):
    pass

sentence2graph_parser = None
bert_tokenizer = None
use_graph = False
use_bpe = True


class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []

        global sentence2graph_parser
        global use_graph
        global use_bpe
        global bert_tokenizer
        if use_graph:
            from text_to_speech.modules.tts.syntaspeech.syntactic_graph_buider import Sentence2GraphParser

        if hparams['ds_name'] in ['libritts', 'librispeech']:
            # Unfortunately, we found when processing libritts with multi-processing will incur pytorch.multiprocessing ERROR
            # so we use single thread with cuda graph builder 
            # it take about 20 hours in a PC with 24-cores-cpu and a RTX2080Ti to process the whole LibriTTS
            # so run the binarization and take a break!
            if use_graph:
                sentence2graph_parser = Sentence2GraphParser("en", use_gpu=True)
            if use_bpe:
                model_name = 'bert-base-uncased'
                tokenizer_kwargs = {'cache_dir': None, 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
                bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        elif hparams['ds_name'] == 'ljspeech':
            # use multi-processing, thus gpu is disabled
            # it takes about 30 minutes for binarization
            if use_graph:
                sentence2graph_parser = Sentence2GraphParser("en", use_gpu=False)
            if use_bpe: 
                model_name = 'bert-base-uncased'
                tokenizer_kwargs = {'cache_dir': None, 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
                bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        elif hparams['preprocess_args']['txt_processor'] == 'zh':
            # use multi-processing, thus gpu is disabled
            # it takes about 30 minutes for binarization
            if use_graph:
                sentence2graph_parser = Sentence2GraphParser("zh", use_gpu=False)
            if use_bpe:
                model_name = 'bert-base-chinese'
                tokenizer_kwargs = {'cache_dir': None, 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
                bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        else:
            pass

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range(self.binarization_args['train_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def valid_item_names(self):
        range_ = self._convert_range(self.binarization_args['valid_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range(self.binarization_args['test_range'])
        return self.item_names[range_[0]:range_[1]]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        for fn in ['phone_set.json', 'word_set.json', 'spk_map.json']:
            remove_file(f"{hparams['binary_data_dir']}/{fn}")
            copy_file(f"{hparams['processed_data_dir']}/{fn}", f"{hparams['binary_data_dir']}/{fn}")
        if hparams['ds_name'] in ['ljspeech', 'biaobei', 'wenetspeech']:
            self.process_data('valid')
            self.process_data('test')
            self.process_data('train')
        elif hparams['ds_name'] in ['libritts', 'librispeech']:
            self.process_data_single_processing('valid')
            self.process_data_single_processing('test')
            self.process_data_single_processing('train')
        else:
            self.process_data('valid')
            self.process_data('test')
            self.process_data('train')
            # raise NotImplementedError

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item, binarization_args=self.binarization_args)
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{'item': item} for item in meta_data]

        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is not None:
                items.append(item)
        if self.binarization_args['with_spk_embed']:
            args = [{'wav': item['wav']} for item in items]
            for item_id, spk_embed in multiprocess_run_tqdm(
                    self.get_spk_embed, args,
                    init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=4,
                    desc='Extracting spk embed'):
                items[item_id]['spk_embed'] = spk_embed

        for item in items:
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_data_single_processing(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        ph_lengths = []
        mel_lengths = []
        total_sec = 0

        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()
        for raw_item in tqdm(meta_data):
            item = self.process_item(raw_item, self.binarization_args)
            if item is None: 
                continue
            if item is not None:
                if use_graph:
                    if item['dgl_graph'].num_nodes() != np.array(item['ph2word']).max():
                        print(f"Skip Item: {item['item_name']} word nodes number incorrect!")
                        continue

            if self.binarization_args['with_spk_embed']:
                spk_embed = self.get_spk_embed(item['wav'],  {'voice_encoder': voice_encoder})
                item['spk_embed'] = spk_embed

            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    # def process_data_single_processing(self, prefix):
    #     data_dir = hparams['binary_data_dir']
    #     builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    #     meta_data = list(self.meta_data(prefix))
    #     ph_lengths = []
    #     mel_lengths = []
    #     total_sec = 0
    #     items = []
    #     args = [{'item': item} for item in meta_data]

    #     for raw_item in tqdm(meta_data):
    #         item = self.process_item(raw_item, self.binarization_args)
    #         if item is not None:
    #             if item['dgl_graph'].num_nodes() != np.array(item['ph2word']).max():
    #                 print(f"Skip Item: {item['item_name']} word nodes number incorrect!")
    #                 continue

    #             items.append(item)

    #     if self.binarization_args['with_spk_embed']:
    #         args = [{'wav': item['wav']} for item in items]
    #         for item_id, spk_embed in multiprocess_run_tqdm(
    #                 self.get_spk_embed, args,
    #                 init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=4,
    #                 desc='Extracting spk embed'):
    #             items[item_id]['spk_embed'] = spk_embed

    #     for item in items:
    #         if not self.binarization_args['with_wav'] and 'wav' in item:
    #             del item['wav']
    #         builder.add_item(item)
    #         mel_lengths.append(item['len'])
    #         assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
    #         if 'ph_len' in item:
    #             ph_lengths.append(item['ph_len'])
    #         total_sec += item['sec']
    #     builder.finalize()
    #     np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
    #     if len(ph_lengths) > 0:
    #         np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
    #     print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args):
        try:
            item['ph_len'] = len(item['ph_token'])
            item_name = item['item_name']
            wav_fn = item['wav_fn']
            wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        except Exception as e:
            print(f"| Skip item ({e}) for index error. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        try:
            n_bos_frames, n_eos_frames = 0, 0
            if binarization_args['with_align']:
                tg_fn = f"{hparams['processed_data_dir']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item)
                if binarization_args['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]

                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * hparams['hop_size']:len(wav) - n_eos_frames * hparams['hop_size']]
            if binarization_args['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None

        # if item['mel'].shape[0] < 64:
        #     print(f"Skip Item: {item['item_name']} Mel-spectrogram is shorter than 64!")
        #     return None
        # fix one bad case of stanza
        if item['txt'].endswith('yn .'):
            item['txt'] = item['txt'][:-4]+'y .'
        if use_graph:
            try:
                language = sentence2graph_parser.language
                if language == 'en':
                    dgl_graph, etypes = sentence2graph_parser.parse(item['txt'])
                elif language == 'zh':
                    dgl_graph, etypes = sentence2graph_parser.parse(item['txt'], item['word'].split(" "), item['ph_gb_word'].split(" "))
                else:
                    raise NotImplementedError
                item['dgl_graph'] = dgl_graph
                item['edge_types'] = etypes
            except:
                print(f"| Dependency Parsing Error! Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
                return None

        if use_bpe:
            sent = item['word'][6:-6] # discard the <BOS> and <EOS>, because the bert_tokenizer cannot recognize them.
            bert_tokens = bert_tokenizer.tokenize(sent)
            input_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
            input_ids.insert(0, 101) # add [CLS] to represent [BOS]
            input_ids.append(102) # add [SEP] to represent [EOS]
            
            bert_tokens.insert(0, '<BOS>')
            bert_tokens.append('<EOS>')
            bert_token2word = []
            word_idx = 0
            for i in range(len(bert_tokens)):
                if not bert_tokens[i].startswith("##"): # this token is a independent word
                    word_idx += 1
                bert_token2word.append(word_idx)

            item['bert_token'] = bert_tokens
            item['bert_input_ids'] = input_ids
            item['bert_token2word'] = bert_token2word
            item['bert_attention_mask'] = [1 for _ in range(len(bert_tokens))]
            item['bert_token_type_ids'] = [0 for _ in range(len(bert_tokens))]

        return item

    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        if binarization_args['with_linear']:
            res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel

    @staticmethod
    def process_align(tg_fn, item):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams['hop_size'], hparams['audio_sample_rate'],
                                     hparams['binarization_args']['min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur

        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch_simple(item['wav'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse
        if hparams['binarization_args']['with_f0cwt']:
            uv, cont_lf0_lpf = get_cont_lf0(f0)
            logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
            cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
            cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
            item['cwt_spec'] = cwt_spec
            item['cwt_mean'] = logf0s_mean_org
            item['cwt_std'] = logf0s_std_org

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))
