import warnings

warnings.filterwarnings("ignore")

import parselmouth
import os
import torch
from skimage.transform import resize
from utils.text_encoder import TokenTextEncoder
from utils.pitch_utils import f0_to_coarse
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
import librosa
import numpy as np
from utils import audio
import pyloudnorm as pyln
import re
import json
from collections import OrderedDict

PUNCS = '!,.?;:'

int16_max = (2 ** 15) - 1


def trim_long_silences(path, sr=None, return_raw_wav=False, norm=True, vad_max_silence_length=12):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :param vad_max_silence_length: Maximum number of consecutive silent frames a segment can have.
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    sampling_rate = 16000
    wav_raw, sr = librosa.core.load(path, sr=sr)

    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav_raw)
        wav_raw = pyln.normalize.loudness(wav_raw, loudness, -20.0)
        if np.abs(wav_raw).max() > 1.0:
            wav_raw = wav_raw / np.abs(wav_raw).max()

    wav = librosa.resample(wav_raw, sr, sampling_rate, res_type='kaiser_best')

    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width = 8

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    audio_mask = resize(audio_mask, (len(wav_raw),)) > 0
    if return_raw_wav:
        return wav_raw, audio_mask, sr
    return wav_raw[audio_mask], audio_mask, sr


def process_utterance(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=22050,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg'):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == 'pwg':
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    else:
        assert False, f'"{vocoder}" is not in ["pwg"].'

    l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    if not return_linear:
        return wav, mel
    else:
        spc = audio.amp_to_db(spc)
        spc = audio.normalize(spc, {'min_level_db': min_level_db})
        return wav, mel, spc


def get_pitch(wav_data, mel, hparams):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
    f0_min = 80
    f0_max = 750

    if hparams['hop_size'] == 128:
        pad_size = 4
    elif hparams['hop_size'] == 256:
        pad_size = 2
    else:
        assert False

    f0 = parselmouth.Sound(wav_data, hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    lpad = pad_size * 2
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    # mel and f0 are extracted by 2 different libraries. we should force them to have the same length.
    # Attention: we find that new version of some libraries could cause ``rpad'' to be a negetive value...
    # Just to be sure, we recommend users to set up the same environments as them in requirements_auto.txt (by Anaconda)
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse


def remove_empty_lines(text):
    """remove empty lines"""
    assert (len(text) > 0)
    assert (isinstance(text, list))
    text = [t.strip() for t in text]
    if "" in text:
        text.remove("")
    return text


class TextGrid(object):
    def __init__(self, text):
        text = remove_empty_lines(text)
        self.text = text
        self.line_count = 0
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        """
        Parameters
        ----------
        pattern : regex to extract pattern
        inc : increment of line count after extraction
        Returns
        -------
        group : extracted info
        """
        try:
            group = re.match(pattern, self.text[self.line_count]).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError("File format error at line %d:%s" % (self.line_count, self.text[self.line_count]))
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError("Only IntervalTier class is supported currently")
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)

    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2)


def get_mel2ph(tg_fn, ph, mel, hparams):
    ph_list = ph.split(" ")
    with open(tg_fn, "r") as f:
        tg = f.readlines()
    tg = remove_empty_lines(tg)
    tg = TextGrid(tg)
    tg = json.loads(tg.toJson())
    split = np.ones(len(ph_list) + 1, np.float) * -1
    tg_idx = 0
    ph_idx = 0
    tg_align = [x for x in tg['tiers'][-1]['items']]
    tg_align_ = []
    for x in tg_align:
        x['xmin'] = float(x['xmin'])
        x['xmax'] = float(x['xmax'])
        if x['text'] in ['sil', 'sp', '', 'SIL', 'PUNC']:
            x['text'] = ''
            if len(tg_align_) > 0 and tg_align_[-1]['text'] == '':
                tg_align_[-1]['xmax'] = x['xmax']
                continue
        tg_align_.append(x)
    tg_align = tg_align_
    tg_len = len([x for x in tg_align if x['text'] != ''])
    ph_len = len([x for x in ph_list if not is_sil_phoneme(x)])
    assert tg_len == ph_len, (tg_len, ph_len, tg_align, ph_list, tg_fn)
    while tg_idx < len(tg_align) or ph_idx < len(ph_list):
        if tg_idx == len(tg_align) and is_sil_phoneme(ph_list[ph_idx]):
            split[ph_idx] = 1e8
            ph_idx += 1
            continue
        x = tg_align[tg_idx]
        if x['text'] == '' and ph_idx == len(ph_list):
            tg_idx += 1
            continue
        assert ph_idx < len(ph_list), (tg_len, ph_len, tg_align, ph_list, tg_fn)
        ph = ph_list[ph_idx]
        if x['text'] == '' and not is_sil_phoneme(ph):
            assert False, (ph_list, tg_align)
        if x['text'] != '' and is_sil_phoneme(ph):
            ph_idx += 1
        else:
            assert (x['text'] == '' and is_sil_phoneme(ph)) \
                   or x['text'].lower() == ph.lower() \
                   or x['text'].lower() == 'sil', (x['text'], ph)
            split[ph_idx] = x['xmin']
            if ph_idx > 0 and split[ph_idx - 1] == -1 and is_sil_phoneme(ph_list[ph_idx - 1]):
                split[ph_idx - 1] = split[ph_idx]
            ph_idx += 1
            tg_idx += 1
    assert tg_idx == len(tg_align), (tg_idx, [x['text'] for x in tg_align])
    assert ph_idx >= len(ph_list) - 1, (ph_idx, ph_list, len(ph_list), [x['text'] for x in tg_align], tg_fn)
    mel2ph = np.zeros([mel.shape[0]], np.int)
    split[0] = 0
    split[-1] = 1e8
    for i in range(len(split) - 1):
        assert split[i] != -1 and split[i] <= split[i + 1], (split[:-1],)
    split = [int(s * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5) for s in split]
    for ph_idx in range(len(ph_list)):
        mel2ph[split[ph_idx]:split[ph_idx + 1]] = ph_idx + 1
    mel2ph_torch = torch.from_numpy(mel2ph)
    T_t = len(ph_list)
    dur = mel2ph_torch.new_zeros([T_t + 1]).scatter_add(0, mel2ph_torch, torch.ones_like(mel2ph_torch))
    dur = dur[1:].numpy()
    return mel2ph, dur


def build_phone_encoder(data_dir):
    phone_list_file = os.path.join(data_dir, 'phone_set.json')
    phone_list = json.load(open(phone_list_file))
    return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')


def is_sil_phoneme(p):
    return not p[0].isalpha()
