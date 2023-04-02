import glob
import re
import librosa
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from modules.parallel_wavegan.models import ParallelWaveGANGenerator
from modules.parallel_wavegan.utils import read_hdf5
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse
from vocoders.base_vocoder import BaseVocoder, register_vocoder
import numpy as np


def load_pwg_model(config_path, checkpoint_path, stats_path):
    # load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ParallelWaveGANGenerator(**config["generator_params"])

    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if 'state_dict' not in ckpt_dict:  # official vocoder
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"]["generator"])
        scaler = StandardScaler()
        if config["format"] == "hdf5":
            scaler.mean_ = read_hdf5(stats_path, "mean")
            scaler.scale_ = read_hdf5(stats_path, "scale")
        elif config["format"] == "npy":
            scaler.mean_ = np.load(stats_path)[0]
            scaler.scale_ = np.load(stats_path)[1]
        else:
            raise ValueError("support only hdf5 or npy format.")
    else:  # custom PWG vocoder
        fake_task = nn.Module()
        fake_task.model_gen = model
        fake_task.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"], strict=False)
        scaler = None

    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| PWG device: {device}.")
    return model, scaler, config, device


@register_vocoder
class PWG(BaseVocoder):
    def __init__(self):
        if hparams['vocoder_ckpt'] == '':  # load LJSpeech PWG pretrained model
            base_dir = 'wavegan_pretrained'
            ckpts = glob.glob(f'{base_dir}/checkpoint-*steps.pkl')
            ckpt = sorted(ckpts, key=
            lambda x: int(re.findall(f'{base_dir}/checkpoint-(\d+)steps.pkl', x)[0]))[-1]
            config_path = f'{base_dir}/config.yaml'
            print('| load PWG: ', ckpt)
            self.model, self.scaler, self.config, self.device = load_pwg_model(
                config_path=config_path,
                checkpoint_path=ckpt,
                stats_path=f'{base_dir}/stats.h5',
            )
        else:
            base_dir = hparams['vocoder_ckpt']
            print(base_dir)
            config_path = f'{base_dir}/config.yaml'
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
            print('| load PWG: ', ckpt)
            self.scaler = None
            self.model, _, self.config, self.device = load_pwg_model(
                config_path=config_path,
                checkpoint_path=ckpt,
                stats_path=f'{base_dir}/stats.h5',
            )

    def spec2wav(self, mel, **kwargs):
        # start generation
        config = self.config
        device = self.device
        pad_size = (config["generator_params"]["aux_context_window"],
                    config["generator_params"]["aux_context_window"])
        c = mel
        if self.scaler is not None:
            c = self.scaler.transform(c)

        with torch.no_grad():
            z = torch.randn(1, 1, c.shape[0] * config["hop_size"]).to(device)
            c = np.pad(c, (pad_size, (0, 0)), "edge")
            c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
            p = kwargs.get('f0')
            if p is not None:
                p = f0_to_coarse(p)
                p = np.pad(p, (pad_size,), "edge")
                p = torch.LongTensor(p[None, :]).to(device)
            y = self.model(z, c, p).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out

    @staticmethod
    def wav2spec(wav_fn, return_linear=False):
        from data_gen.tts.data_gen_utils import process_utterance
        res = process_utterance(
            wav_fn, fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=return_linear, vocoder='pwg', eps=float(hparams.get('wav2spec_eps', 1e-10)))
        if return_linear:
            return res[0], res[1].T, res[2].T  # [T, 80], [T, n_fft]
        else:
            return res[0], res[1].T

    @staticmethod
    def wav2mfcc(wav_fn):
        fft_size = hparams['fft_size']
        hop_size = hparams['hop_size']
        win_length = hparams['win_size']
        sample_rate = hparams['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13,
                                    n_fft=fft_size, hop_length=hop_size,
                                    win_length=win_length, pad_mode="constant", power=1.0)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta]).T
        return mfcc
