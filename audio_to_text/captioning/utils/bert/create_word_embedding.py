# -*- coding: utf-8 -*-

import sys
import os

from bert_serving.client import BertClient
import numpy as np
from tqdm import tqdm
import fire
import torch

sys.path.append(os.getcwd())
from utils.build_vocab import Vocabulary

def main(vocab_file: str, output: str, server_hostname: str):
    client = BertClient(ip=server_hostname)
    vocabulary = torch.load(vocab_file)
    vocab_size = len(vocabulary)
    
    fake_embedding = client.encode(["test"]).reshape(-1)
    embed_size = fake_embedding.shape[0]

    print("Encoding words into embeddings with size: ", embed_size)

    embeddings = np.empty((vocab_size, embed_size))
    for i in tqdm(range(len(embeddings)), ascii=True):
        embeddings[i] = client.encode([vocabulary.idx2word[i]])
    np.save(output, embeddings)


if __name__ == '__main__':
    fire.Fire(main)

    
