import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from h5py import File
import sklearn.metrics

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("train_feature", type=str)
parser.add_argument("train_corpus", type=str)
parser.add_argument("pred_feature", type=str)
parser.add_argument("output_json", type=str)

args = parser.parse_args()
train_embs = []
train_idx_to_audioid = []
with File(args.train_feature, "r") as store:
    for audio_id, embedding in tqdm(store.items(), ascii=True):
        train_embs.append(embedding[()])
        train_idx_to_audioid.append(audio_id)

train_annotation = json.load(open(args.train_corpus, "r"))["audios"]
train_audioid_to_tokens = {}
for item in train_annotation:
    audio_id = item["audio_id"]
    train_audioid_to_tokens[audio_id] = [cap_item["tokens"] for cap_item in item["captions"]]
train_embs = np.stack(train_embs)


pred_data = []
pred_embs = []
pred_idx_to_audioids = []
with File(args.pred_feature, "r") as store:
    for audio_id, embedding in tqdm(store.items(), ascii=True):
        pred_embs.append(embedding[()])
        pred_idx_to_audioids.append(audio_id)
pred_embs = np.stack(pred_embs)

similarity = sklearn.metrics.pairwise.cosine_similarity(pred_embs, train_embs)
for idx, audio_id in enumerate(pred_idx_to_audioids):
    train_idx = similarity[idx].argmax()
    pred_data.append({
        "filename": audio_id,
        "tokens": random.choice(train_audioid_to_tokens[train_idx_to_audioid[train_idx]])
    })
json.dump({"predictions": pred_data}, open(args.output_json, "w"), ensure_ascii=False, indent=4)
