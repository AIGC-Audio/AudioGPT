import copy
import json

import numpy as np
import fire


def evaluate_annotation(key2refs, scorer):
    if scorer.method() == "Bleu":
        scores = np.array([ 0.0 for n in range(4) ])
    else:
        scores = 0
    num_cap_per_audio = len(next(iter(key2refs.values())))

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in key2refs:
                key2refs[key].insert(0, res[key][0])
        res = { key: [refs.pop(),] for key, refs in key2refs.items() }
        score, _ = scorer.compute_score(key2refs, res)
        
        if scorer.method() == "Bleu":
            scores += np.array(score)
        else:
            scores += score
    
    score = scores / num_cap_per_audio
    return score
   
def evaluate_prediction(key2pred, key2refs, scorer):
    if scorer.method() == "Bleu":
        scores = np.array([ 0.0 for n in range(4) ])
    else:
        scores = 0
    num_cap_per_audio = len(next(iter(key2refs.values())))

    for i in range(num_cap_per_audio):
        key2refs_i = {}
        for key, refs in key2refs.items():
            key2refs_i[key] = refs[:i] + refs[i+1:]
        score, _ = scorer.compute_score(key2refs_i, key2pred)
        
        if scorer.method() == "Bleu":
            scores += np.array(score)
        else:
            scores += score
    
    score = scores / num_cap_per_audio
    return score


class Evaluator(object):

    def eval_annotation(self, annotation, output):
        captions = json.load(open(annotation, "r"))["audios"]

        key2refs = {}
        for audio_idx in range(len(captions)):
            audio_id = captions[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in captions[audio_idx]["captions"]:
                key2refs[audio_id].append(caption["caption"])

        from fense.fense import Fense
        scores = {}
        scorer = Fense()
        scores[scorer.method()] = evaluate_annotation(copy.deepcopy(key2refs), scorer)

        refs4eval = {}
        for key, refs in key2refs.items():
            refs4eval[key] = []
            for idx, ref in enumerate(refs):
                refs4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": ref
                })

        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        key2refs = tokenizer.tokenize(refs4eval)


        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        

        scorers = [Bleu(), Rouge(), Cider(), Meteor(), Spice()]
        for scorer in scorers:
            scores[scorer.method()] = evaluate_annotation(copy.deepcopy(key2refs), scorer)

        spider = 0
        with open(output, "w") as f:
            for name, score in scores.items():
                if name == "Bleu":
                    for n in range(4):
                        f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))
                else:
                    f.write("{}: {:6.3f}\n".format(name, score))
                    if name in ["CIDEr", "SPICE"]:
                        spider += score
            f.write("SPIDEr: {:6.3f}\n".format(spider / 2))

    def eval_prediction(self, prediction, annotation, output):
        ref_captions = json.load(open(annotation, "r"))["audios"]

        key2refs = {}
        for audio_idx in range(len(ref_captions)):
            audio_id = ref_captions[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in ref_captions[audio_idx]["captions"]:
                key2refs[audio_id].append(caption["caption"])

        pred_captions = json.load(open(prediction, "r"))["predictions"]

        key2pred = {}
        for audio_idx in range(len(pred_captions)):
            item = pred_captions[audio_idx]
            audio_id = item["filename"]
            key2pred[audio_id] = [item["tokens"]]

        from fense.fense import Fense
        scores = {}
        scorer = Fense()
        scores[scorer.method()] = evaluate_prediction(key2pred, key2refs, scorer)

        refs4eval = {}
        for key, refs in key2refs.items():
            refs4eval[key] = []
            for idx, ref in enumerate(refs):
                refs4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": ref
                })

        preds4eval = {}
        for key, preds in key2pred.items():
            preds4eval[key] = []
            for idx, pred in enumerate(preds):
                preds4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": pred
                })

        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        key2refs = tokenizer.tokenize(refs4eval)
        key2pred = tokenizer.tokenize(preds4eval)


        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice

        scorers = [Bleu(), Rouge(), Cider(), Meteor(), Spice()]
        for scorer in scorers:
            scores[scorer.method()] = evaluate_prediction(key2pred, key2refs, scorer)

        spider = 0
        with open(output, "w") as f:
            for name, score in scores.items():
                if name == "Bleu":
                    for n in range(4):
                        f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))
                else:
                    f.write("{}: {:6.3f}\n".format(name, score))
                    if name in ["CIDEr", "SPICE"]:
                        spider += score
            f.write("SPIDEr: {:6.3f}\n".format(spider / 2))


if __name__ == "__main__":
    fire.Fire(Evaluator)
