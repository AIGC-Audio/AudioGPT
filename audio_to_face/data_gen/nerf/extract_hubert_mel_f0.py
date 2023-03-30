import soundfile as sf
import numpy as np
import torch
from argparse import ArgumentParser
from audio_to_face.data_gen.process_lrs3.process_audio_hubert import get_hubert_from_16k_speech
from audio_to_face.data_gen.process_lrs3.process_audio_mel_f0 import extract_mel_f0_from_fname

parser = ArgumentParser()
parser.add_argument('--video_id', type=str, default='May', help='')
args = parser.parse_args()

person_id = args.video_id
wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
hubert_npy_name = f"data/processed/videos/{person_id}/aud_hubert.npy"
mel_f0_npy_name = f"data/processed/videos/{person_id}/aud_mel_f0.npy"
speech_16k, _ = sf.read(wav_16k_name)
hubert_hidden = get_hubert_from_16k_speech(speech_16k)
np.save(hubert_npy_name, hubert_hidden.detach().numpy())
print(f"Hubert extracted at {hubert_npy_name}")
extract_mel_f0_from_fname(wav_16k_name, out_name=mel_f0_npy_name)
print(f"Mel and F0 extracted at {mel_f0_npy_name}")
