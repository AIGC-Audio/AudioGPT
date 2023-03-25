import os
import json
import tqdm

from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from functools import partial

# def process_segment0(segment, opus_path, audio_out_dir, audio_id):
#     segment_id = segment['sid']
#     item_name = segment_id
#     begin_time = segment['begin_time']
#     end_time = segment['end_time']
#     out_wav_path = os.path.join(audio_out_dir, segment_id+'.wav')
#     text = segment['text_tn']
#     text = text.replace("<COMMA>", ",")
#     text = text.replace("<PERIOD>", ".")
#     text = text.replace("<QUESTIONMARK>", "?")
#     text = text.replace("<EXCLAMATIONPOINT>", "!")
#     text = text.lower()
#     item_meta = {'item_name': item_name, 'wav_fn': out_wav_path, 'txt': text, 'spk_name': audio_id}
#     return item_meta

def process_segment(segment, opus_path, audio_out_dir, audio_id):
    segment_id = segment['sid']
    item_name = segment_id
    begin_time = segment['begin_time']
    end_time = segment['end_time']
    out_wav_path = os.path.join(audio_out_dir, segment_id+'.wav')
    if os.path.exists(out_wav_path):
        return
    cmd = f'ffmpeg -v quiet -y -i {opus_path} -ac 1 -ar 16000 -ss {begin_time} -to {end_time} {out_wav_path}'
    os.system(cmd)
    text = segment['text_tn']
    text = text.replace("<COMMA>", ",")
    text = text.replace("<PERIOD>", ".")
    text = text.replace("<QUESTIONMARK>", "?")
    text = text.replace("<EXCLAMATIONPOINT>", "!")
    text = text.lower()
    item_meta = {'item_name': item_name, 'wav_fn': out_wav_path, 'txt': text, 'spk_name': audio_id}
    return item_meta

giga_root_dir = '/home/yezhenhui/datasets/raw/GigaSpeech/'
giga_out_dir = '/home/yezhenhui/datasets/raw/GigaSpeech_extract/'
os.makedirs(giga_out_dir, exist_ok=True)

with open(f'{giga_root_dir}/GigaSpeech.json', 'r') as injson:
    json_data = json.load(injson)

meta = []
out_meta_name = os.path.join(giga_out_dir, 'meta.json')

audio_corpus = json_data['audios'] # list of dict, length 38131

args = []
for audio_source in tqdm.tqdm(audio_corpus, total=len(audio_corpus), desc='loading the args'):
    audio_id = audio_source['aid']
    subset = audio_source['subsets']
    audio_path = audio_source['path']
    opus_path = os.path.join(giga_root_dir, audio_path)
    audio_out_dir = os.path.join(giga_out_dir, os.path.dirname(audio_path), audio_id)
    os.makedirs(audio_out_dir, exist_ok=True)
    segments = audio_source['segments']
    spk_name = audio_id
    args += [{'segment': segment, 'opus_path': opus_path, 'audio_out_dir': audio_out_dir, 'audio_id': audio_id} for segment in segments]

# for segment_meta in multiprocess_run_tqdm(process_segment0, args, desc='extracting...'):
#     meta += segment_meta

# with open(out_meta_name, 'w') as f:
#     json.dump(meta, f)
# print("successful!")

for segment_meta in multiprocess_run_tqdm(process_segment, args, num_workers=32, desc='extracting...'):
   pass


