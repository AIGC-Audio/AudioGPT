import os
import random
from copy import deepcopy
import pandas as pd
import logging
from tqdm import tqdm
import json
import glob
import re
from resemblyzer import VoiceEncoder
import traceback
import numpy as np
import pretty_midi
import librosa
from scipy.interpolate import interp1d
import torch
from textgrid import TextGrid

from utils.hparams import hparams
from data_gen.tts.data_gen_utils import build_phone_encoder, get_pitch
from utils.pitch_utils import f0_to_coarse
from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.binarizer_zh import ZhBinarizer
from data_gen.tts.txt_processors.zh_g2pM import ALL_YUNMU
from vocoders.base_vocoder import VOCODERS


class SingingBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2f0fn = {}
        self.item2tgfn = {}
        self.item2spk = {}

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            wav_suffix = '_wf0.wav'
            txt_suffix = '.txt'
            ph_suffix = '_ph.txt'
            tg_suffix = '.TextGrid'
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*/*{wav_suffix}')

            for piece_path in all_wav_pieces:
                item_name = raw_item_name = piece_path[len(processed_data_dir)+1:].replace('/', '-')[:-len(wav_suffix)]
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = open(f'{piece_path.replace(wav_suffix, txt_suffix)}').readline()
                self.item2ph[item_name] = open(f'{piece_path.replace(wav_suffix, ph_suffix)}').readline()
                self.item2wavfn[item_name] = piece_path

                self.item2spk[item_name] = re.split('-|#', piece_path.split('/')[-2])[0]
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
                self.item2tgfn[item_name] = piece_path.replace(wav_suffix, tg_suffix)
        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.phone_encoder = self._phone_encoder()
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for ph_sent in self.item2ph.values():
                ph_set += ph_sent.split(' ')
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])

    # @staticmethod
    # def get_pitch(wav_fn, spec, res):
    #     wav_suffix = '_wf0.wav'
    #     f0_suffix = '_f0.npy'
    #     f0fn = wav_fn.replace(wav_suffix, f0_suffix)
    #     pitch_info = np.load(f0fn)
    #     f0 = [x[1] for x in pitch_info]
    #     spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
    #     f0_x_coor = np.arange(0, 1, 1 / len(f0))[:len(f0)]
    #     f0 = interp1d(f0_x_coor, f0, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]
    #     # f0_x_coor = np.arange(0, 1, 1 / len(f0))
    #     # f0_x_coor[-1] = 1
    #     # f0 = interp1d(f0_x_coor, f0, 'nearest')(spec_x_coor)[:len(spec)]
    #     if sum(f0) == 0:
    #         raise BinarizationError("Empty f0")
    #     assert len(f0) == len(spec), (len(f0), len(spec))
    #     pitch_coarse = f0_to_coarse(f0)
    #
    #     # vis f0
    #     # import matplotlib.pyplot as plt
    #     # from textgrid import TextGrid
    #     # tg_fn = wav_fn.replace(wav_suffix, '.TextGrid')
    #     # fig = plt.figure(figsize=(12, 6))
    #     # plt.pcolor(spec.T, vmin=-5, vmax=0)
    #     # ax = plt.gca()
    #     # ax2 = ax.twinx()
    #     # ax2.plot(f0, color='red')
    #     # ax2.set_ylim(0, 800)
    #     # itvs = TextGrid.fromFile(tg_fn)[0]
    #     # for itv in itvs:
    #     #     x = itv.maxTime * hparams['audio_sample_rate'] / hparams['hop_size']
    #     #     plt.vlines(x=x, ymin=0, ymax=80, color='black')
    #     #     plt.text(x=x, y=20, s=itv.mark, color='black')
    #     # plt.savefig('tmp/20211229_singing_plots_test.png')
    #
    #     res['f0'] = f0
    #     res['pitch'] = pitch_coarse

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                # cls.get_pitch(wav_fn, mel, res)
                cls.get_pitch(wav, mel, res)
            if binarization_args['with_txt']:
                try:
                    # print(ph)
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(tg_fn, ph, mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


class MidiSingingBinarizer(SingingBinarizer):
    item2midi = {}
    item2midi_dur = {}
    item2is_slur = {}
    item2ph_durs = {}
    item2wdb = {}

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            meta_midi = json.load(open(os.path.join(processed_data_dir, 'meta.json')))   # [list of dict]

            for song_item in meta_midi:
                item_name = raw_item_name = song_item['item_name']
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2wavfn[item_name] = song_item['wav_fn']
                self.item2txt[item_name] = song_item['txt']

                self.item2ph[item_name] = ' '.join(song_item['phs'])
                self.item2wdb[item_name] = [1 if x in ALL_YUNMU + ['AP', 'SP', '<SIL>'] else 0 for x in song_item['phs']]
                self.item2ph_durs[item_name] = song_item['ph_dur']

                self.item2midi[item_name] = song_item['notes']
                self.item2midi_dur[item_name] = song_item['notes_dur']
                self.item2is_slur[item_name] = song_item['is_slur']
                self.item2spk[item_name] = 'pop-cs'
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"

        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        wav_suffix = '.wav'
        # midi_suffix = '.mid'
        wav_dir = 'wavs'
        f0_dir = 'f0'

        item_name = '/'.join(os.path.splitext(wav_fn)[0].split('/')[-2:]).replace('_wf0', '')
        res['pitch_midi'] = np.asarray(MidiSingingBinarizer.item2midi[item_name])
        res['midi_dur'] = np.asarray(MidiSingingBinarizer.item2midi_dur[item_name])
        res['is_slur'] = np.asarray(MidiSingingBinarizer.item2is_slur[item_name])
        res['word_boundary'] = np.asarray(MidiSingingBinarizer.item2wdb[item_name])
        assert res['pitch_midi'].shape == res['midi_dur'].shape == res['is_slur'].shape, (
        res['pitch_midi'].shape, res['midi_dur'].shape, res['is_slur'].shape)

        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @staticmethod
    def get_align(ph_durs, mel, phone_encoded, res, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        # print('ph durs: ', ph_durs)
        # print('mel2ph: ', mel2ph, len(mel2ph))
        res['mel2ph'] = mel2ph
        # res['dur'] = None

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(MidiSingingBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


class ZhSingingBinarizer(ZhBinarizer, SingingBinarizer):
    pass


class OpencpopBinarizer(MidiSingingBinarizer):
    item2midi = {}
    item2midi_dur = {}
    item2is_slur = {}
    item2ph_durs = {}
    item2wdb = {}

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        raw_data_dir = hparams['raw_data_dir']
        # meta_midi = json.load(open(os.path.join(raw_data_dir, 'meta.json')))   # [list of dict]
        utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt')).readlines()

        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            item_name = raw_item_name = song_info[0]
            self.item2wavfn[item_name] = f'{raw_data_dir}/wavs/{item_name}.wav'
            self.item2txt[item_name] = song_info[1]

            self.item2ph[item_name] = song_info[2]
            # self.item2wdb[item_name] = list(np.nonzero([1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()])[0])
            self.item2wdb[item_name] = [1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()]
            self.item2ph_durs[item_name] = [float(x) for x in song_info[5].split(" ")]

            self.item2midi[item_name] = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                   for x in song_info[3].split(" ")]
            self.item2midi_dur[item_name] = [float(x) for x in song_info[4].split(" ")]
            self.item2is_slur[item_name] = [int(x) for x in song_info[6].split(" ")]
            self.item2spk[item_name] = 'opencpop'

        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        wav_suffix = '.wav'
        # midi_suffix = '.mid'
        wav_dir = 'wavs'
        f0_dir = 'text_f0_align'

        item_name = os.path.splitext(os.path.basename(wav_fn))[0]
        res['pitch_midi'] = np.asarray(OpencpopBinarizer.item2midi[item_name])
        res['midi_dur'] = np.asarray(OpencpopBinarizer.item2midi_dur[item_name])
        res['is_slur'] = np.asarray(OpencpopBinarizer.item2is_slur[item_name])
        res['word_boundary'] = np.asarray(OpencpopBinarizer.item2wdb[item_name])
        assert res['pitch_midi'].shape == res['midi_dur'].shape == res['is_slur'].shape, (res['pitch_midi'].shape, res['midi_dur'].shape, res['is_slur'].shape)

        # gt f0.
        # f0 = None
        # f0_suffix = '_f0.npy'
        # f0fn = wav_fn.replace(wav_suffix, f0_suffix).replace(wav_dir, f0_dir)
        # pitch_info = np.load(f0fn)
        # f0 = [x[1] for x in pitch_info]
        # spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
        #
        # f0_x_coor = np.arange(0, 1, 1 / len(f0))[:len(f0)]
        # f0 = interp1d(f0_x_coor, f0, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]
        # if sum(f0) == 0:
        #     raise BinarizationError("Empty **gt** f0")
        #
        # pitch_coarse = f0_to_coarse(f0)
        # res['f0'] = f0
        # res['pitch'] = pitch_coarse

        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(OpencpopBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


if __name__ == "__main__":
    SingingBinarizer().process()
