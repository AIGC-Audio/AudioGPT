import pickle
import fire
import pandas as pd
from tqdm import tqdm
from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer
import torch
from h5py import File


class EmbeddingExtractor(object):

    def extract(self, caption_file: str, model, output: str):
        caption_df = pd.read_json(caption_file, dtype={"key": str})
        embeddings = {}
        with tqdm(total=caption_df.shape[0], ascii=True) as pbar:
            for idx, row in caption_df.iterrows():
                key = row["key"]
                caption = row["caption"]
                value = model.encode([caption])[0]
                if key not in embeddings:
                    embeddings[key] = [value]
                else:
                    embeddings[key].append(value)
                pbar.update()

        dump = {}
        for key in embeddings:
            dump[key] = torch.stack(embeddings[key]).numpy()

        with open(output, "wb") as f:
            pickle.dump(dump, f)

    def extract_sentbert(self, caption_file: str, output: str, zh: bool=False):
        lang2model = {
            "zh": "distiluse-base-multilingual-cased",
            "en": "bert-base-nli-mean-tokens"
        }
        lang = "zh" if zh else "en"
        model = SentenceTransformer(lang2model[lang])
        self.extract(caption_file, model, output)

    def extract_originbert(self, caption_file: str, output: str, ip="localhost"):
        client = BertClient(ip)
        model = lambda captions: client.encode(captions)
        self.extract(caption_file, model, output)

    def extract_sbert(self, input_json: str, output: str):
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        data = pd.read_json(input_json)["audios"]
        with tqdm(total=data.shape[0], ascii=True) as pbar, File(output, "w") as store:
            for idx, sample in data.iterrows():
                audio_id = sample["audio_id"]
                for cap in sample["captions"]:
                    cap_id = cap["cap_id"]
                    store[f"{audio_id}_{cap_id}"] = model.encode(cap["caption"])
                pbar.update()


if __name__ == "__main__":
    fire.Fire(EmbeddingExtractor)
