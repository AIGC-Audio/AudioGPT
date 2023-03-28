import sys
import os
import librosa
import numpy as np
import torch
import audio_to_text.captioning.models
import audio_to_text.captioning.models.encoder
import audio_to_text.captioning.models.decoder
import audio_to_text.captioning.utils.train_util as train_util


def load_model(config, checkpoint):
    ckpt = torch.load(checkpoint, "cpu")
    encoder_cfg = config["model"]["encoder"]
    encoder = train_util.init_obj(
        audio_to_text.captioning.models.encoder,
        encoder_cfg
    )
    if "pretrained" in encoder_cfg:
        pretrained = encoder_cfg["pretrained"]
        train_util.load_pretrained_model(encoder,
                                         pretrained,
                                         sys.stdout.write)
    decoder_cfg = config["model"]["decoder"]
    if "vocab_size" not in decoder_cfg["args"]:
        decoder_cfg["args"]["vocab_size"] = len(ckpt["vocabulary"])
    decoder = train_util.init_obj(
        audio_to_text.captioning.models.decoder,
        decoder_cfg
    )
    if "word_embedding" in decoder_cfg:
        decoder.load_word_embedding(**decoder_cfg["word_embedding"])
    if "pretrained" in decoder_cfg:
        pretrained = decoder_cfg["pretrained"]
        train_util.load_pretrained_model(decoder,
                                         pretrained,
                                         sys.stdout.write)
    model = train_util.init_obj(audio_to_text.captioning.models, config["model"],
        encoder=encoder, decoder=decoder)
    train_util.load_pretrained_model(model, ckpt)
    model.eval()
    return {
        "model": model,
        "vocabulary": ckpt["vocabulary"]
    }


def decode_caption(word_ids, vocabulary):
    candidate = []
    for word_id in word_ids:
        word = vocabulary[word_id]
        if word == "<end>":
            break
        elif word == "<start>":
            continue
        candidate.append(word)
    candidate = " ".join(candidate)
    return candidate


class AudioCapModel(object):
    def __init__(self,weight_dir,device='cuda'):
        config = os.path.join(weight_dir,'config.yaml')
        self.config = train_util.parse_config_or_kwargs(config)
        checkpoint = os.path.join(weight_dir,'swa.pth')
        resumed = load_model(self.config, checkpoint)
        model = resumed["model"]
        self.vocabulary = resumed["vocabulary"]
        self.model = model.to(device)
        self.device = device

    def caption(self,audio_list):
        if isinstance(audio_list,np.ndarray):
            audio_list = [audio_list]
        elif isinstance(audio_list,str):
            audio_list = [librosa.load(audio_list,sr=32000)[0]]
        
        captions = []
        for wav in audio_list:
            inputwav = torch.as_tensor(wav).float().unsqueeze(0).to(self.device)
            wav_len = torch.LongTensor([len(wav)])
            input_dict = {
                "mode": "inference",
                "wav": inputwav,
                "wav_len": wav_len,
                "specaug": False,
                "sample_method": "beam",
            }
            print(input_dict)
            out_dict = self.model(input_dict)
            caption_batch = [decode_caption(seq, self.vocabulary) for seq in \
                out_dict["seq"].cpu().numpy()]
            captions.extend(caption_batch)
        return captions


        
    def __call__(self, audio_list):
        return self.caption(audio_list)



