#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import math
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
import argparse
import os
import sys
import glob
from sklearn import metrics
import soundfile as sf
#import sentencepiece as spm
import torch
import inference as encoder
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from resemblyzer import VoiceEncoder, preprocess_wav


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);

    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);

    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE])

    return (tunedThreshold, eer, fpr, fnr);


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio,sample_rate = sf.read(filename)

    feats_v0 = torch.from_numpy(audio).float()
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = math.floor((max_audio - audiosize + 1) / 2)
        audio = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
    feat = numpy.stack(feats, axis=0)
    feat = torch.FloatTensor(feat)
    return feat;

def evaluateFromList(listfilename, print_interval=100, test_path='', multi=False):

    lines       = []
    files       = []
    feats       = {}
    tstart      = time.time()

    ## Read all lines
    with open(listfilename) as listfile:
        while True:
            line = listfile.readline();
            if (not line):
                break;

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            files.append(data[1])
            files.append(data[2])
            lines.append(line)

    setfiles = list(set(files))
    setfiles.sort()
    ## Save all features to file
    for idx, file in enumerate(setfiles):
        # preprocessed_wav = encoder.preprocess_wav(os.path.join(test_path,file))
        # embed = encoder.embed_utterance(preprocessed_wav)
        processed_wav = preprocess_wav(os.path.join(test_path,file))
        embed = voice_encoder.embed_utterance(processed_wav)

        torch.cuda.empty_cache()
        ref_feat = torch.from_numpy(embed).unsqueeze(0)

        feats[file]     = ref_feat

        telapsed = time.time() - tstart

        if idx % print_interval == 0:
            sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

    print('')
    all_scores = [];
    all_labels = [];
    all_trials = [];
    tstart = time.time()

    ## Read files and compute all scores
    for idx, line in enumerate(lines):

        data = line.split();
        ## Append random label if missing
        if len(data) == 2: data = [random.randint(0,1)] + data

        ref_feat = feats[data[1]]
        com_feat = feats[data[2]]
        ref_feat = ref_feat.cuda()
        com_feat = com_feat.cuda()
        # normalize feats
        ref_feat = F.normalize(ref_feat, p=2, dim=1)
        com_feat = F.normalize(com_feat, p=2, dim=1)

        dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1)).detach().cpu().numpy();

        score = -1 * numpy.mean(dist);

        all_scores.append(score);
        all_labels.append(int(data[0]));
        all_trials.append(data[1]+" "+data[2])

        if idx % print_interval == 0:
            telapsed = time.time() - tstart
            sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
            sys.stdout.flush();

    print('\n')

    return (all_scores, all_labels, all_trials);



if __name__ == '__main__':

    parser = argparse.ArgumentParser("baseline")
    parser.add_argument("--data_root", type=str, help="", required=True)
    parser.add_argument("--list", type=str, help="", required=True)
    parser.add_argument("--model_dir", type=str, help="model parameters for AudioEncoder", required=True)

    args = parser.parse_args()


    # Load the models one by one.
    print("Preparing the encoder...")
    # encoder.load_model(Path(args.model_dir))
    print("Insert the wav file name...")
    voice_encoder = VoiceEncoder().cuda()

    sc, lab, trials = evaluateFromList(args.list, print_interval=100, test_path=args.data_root)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])
