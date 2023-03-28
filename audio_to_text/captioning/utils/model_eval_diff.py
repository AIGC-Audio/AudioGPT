import os
import sys
import copy
import pickle

import numpy as np
import pandas as pd
import fire

sys.path.append(os.getcwd())


def coco_score(refs, pred, scorer):
    if scorer.method() == "Bleu":
        scores = np.array([ 0.0 for n in range(4) ])
    else:
        scores = 0
    num_cap_per_audio = len(refs[list(refs.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in refs:
                refs[key].insert(0, res[key][0])
        res = {key: [refs[key].pop(),] for key in refs}
        score, _ = scorer.compute_score(refs, pred)    
        
        if scorer.method() == "Bleu":
            scores += np.array(score)
        else:
            scores += score
    
    score = scores / num_cap_per_audio

    for key in refs:
        refs[key].insert(0, res[key][0])
    score_allref, _ = scorer.compute_score(refs, pred)
    diff = score_allref - score
    return diff

def embedding_score(refs, pred, scorer):

    num_cap_per_audio = len(refs[list(refs.keys())[0]])
    scores = 0

    for i in range(num_cap_per_audio):
        res = {key: [refs[key][i],] for key in refs.keys() if len(refs[key]) == num_cap_per_audio}
        refs_i = {key: np.concatenate([refs[key][:i], refs[key][i+1:]]) for key in refs.keys() if len(refs[key]) == num_cap_per_audio}
        score, _ = scorer.compute_score(refs_i, pred)    
        
        scores += score
    
    score = scores / num_cap_per_audio

    score_allref, _ = scorer.compute_score(refs, pred)
    diff = score_allref - score
    return diff
   
def main(output_file, eval_caption_file, eval_embedding_file, output, zh=False):
    output_df = pd.read_json(output_file)
    output_df["key"] = output_df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    pred = output_df.groupby("key")["tokens"].apply(list).to_dict()

    label_df = pd.read_json(eval_caption_file)
    if zh:
        refs = label_df.groupby("key")["tokens"].apply(list).to_dict()
    else:
        refs = label_df.groupby("key")["caption"].apply(list).to_dict()

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    scorer = Bleu(zh=zh)
    bleu_scores = coco_score(copy.deepcopy(refs), pred, scorer)
    scorer = Cider(zh=zh)
    cider_score = coco_score(copy.deepcopy(refs), pred, scorer)
    scorer = Rouge(zh=zh)
    rouge_score = coco_score(copy.deepcopy(refs), pred, scorer)

    if not zh:
        from pycocoevalcap.meteor.meteor import Meteor
        scorer = Meteor()
        meteor_score = coco_score(copy.deepcopy(refs), pred, scorer)

        from pycocoevalcap.spice.spice import Spice
        scorer = Spice()
        spice_score = coco_score(copy.deepcopy(refs), pred, scorer)
    
    # from audiocaptioneval.sentbert.sentencebert import SentenceBert
    # scorer = SentenceBert(zh=zh)
    # with open(eval_embedding_file, "rb") as f:
        # ref_embeddings = pickle.load(f)

    # sent_bert = embedding_score(ref_embeddings, pred, scorer)

    with open(output, "w") as f:
        f.write("Diff:\n")
        for n in range(4):
            f.write("BLEU-{}: {:6.3f}\n".format(n+1, bleu_scores[n]))
        f.write("CIDEr: {:6.3f}\n".format(cider_score))
        f.write("ROUGE: {:6.3f}\n".format(rouge_score))
        if not zh:
            f.write("Meteor: {:6.3f}\n".format(meteor_score))
            f.write("SPICE: {:6.3f}\n".format(spice_score))
        # f.write("SentenceBert: {:6.3f}\n".format(sent_bert))



if __name__ == "__main__":
    fire.Fire(main)
