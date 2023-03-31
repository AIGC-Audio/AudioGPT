# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn

from .base_model import CaptionModel
from .utils import repeat_tensor
import audio_to_text.captioning.models.decoder


class TransformerModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                audio_to_text.captioning.models.decoder.TransformerDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)

    def seq_forward(self, input_dict):
        cap = input_dict["cap"]
        cap_padding_mask = (cap == self.pad_idx).to(cap.device)
        cap_padding_mask = cap_padding_mask[:, :-1]
        output = self.decoder(
            {
                "word": cap[:, :-1],
                "attn_emb": input_dict["attn_emb"],
                "attn_emb_len": input_dict["attn_emb_len"],
                "cap_padding_mask": cap_padding_mask
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_emb": input_dict["attn_emb"],
            "attn_emb_len": input_dict["attn_emb_len"]
        }
        t = input_dict["t"]
        
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["cap"][:, :t+1]
        else:
            start_word = torch.tensor([self.start_idx,] * input_dict["attn_emb"].size(0)).unsqueeze(1).long()
            if t == 0:
                word = start_word
            else:
                word = torch.cat((start_word, output["seq"][:, :t]), dim=-1)
        # word: [N, T]
        decoder_input["word"] = word

        cap_padding_mask = (word == self.pad_idx).to(input_dict["attn_emb"].device)
        decoder_input["cap_padding_mask"] = cap_padding_mask
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_emb = repeat_tensor(input_dict["attn_emb"][i], beam_size)
            attn_emb_len = repeat_tensor(input_dict["attn_emb_len"][i], beam_size)
            output_i["attn_emb"] = attn_emb
            output_i["attn_emb_len"] = attn_emb_len
        decoder_input["attn_emb"] = output_i["attn_emb"]
        decoder_input["attn_emb_len"] = output_i["attn_emb_len"]
        ###############
        # determine input word
        ################
        start_word = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long()
        if t == 0:
            word = start_word
        else:
            word = torch.cat((start_word, output_i["seq"]), dim=-1)
        decoder_input["word"] = word
        cap_padding_mask = (word == self.pad_idx).to(input_dict["attn_emb"].device)
        decoder_input["cap_padding_mask"] = cap_padding_mask

        return decoder_input


class M2TransformerModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.M2TransformerDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
        self.check_encoder_compatibility()

    def check_encoder_compatibility(self):
        assert isinstance(self.encoder, captioning.models.encoder.M2TransformerEncoder), \
            f"only M2TransformerModel is compatible with {self.__class__.__name__}"


    def seq_forward(self, input_dict):
        cap = input_dict["cap"]
        output = self.decoder(
            {
                "word": cap[:, :-1],
                "attn_emb": input_dict["attn_emb"],
                "attn_emb_mask": input_dict["attn_emb_mask"],
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_emb": input_dict["attn_emb"],
            "attn_emb_mask": input_dict["attn_emb_mask"]
        }
        t = input_dict["t"]
        
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["cap"][:, :t+1]
        else:
            start_word = torch.tensor([self.start_idx,] * input_dict["attn_emb"].size(0)).unsqueeze(1).long()
            if t == 0:
                word = start_word
            else:
                word = torch.cat((start_word, output["seq"][:, :t]), dim=-1)
        # word: [N, T]
        decoder_input["word"] = word

        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_emb = repeat_tensor(input_dict["attn_emb"][i], beam_size)
            attn_emb_mask = repeat_tensor(input_dict["attn_emb_mask"][i], beam_size)
            output_i["attn_emb"] = attn_emb
            output_i["attn_emb_mask"] = attn_emb_mask
        decoder_input["attn_emb"] = output_i["attn_emb"]
        decoder_input["attn_emb_mask"] = output_i["attn_emb_mask"]
        ###############
        # determine input word
        ################
        start_word = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long()
        if t == 0:
            word = start_word
        else:
            word = torch.cat((start_word, output_i["seq"]), dim=-1)
        decoder_input["word"] = word

        return decoder_input


class EventEncoder(nn.Module):
    """
    Encode the Label information in AudioCaps and AudioSet
    """
    def __init__(self, emb_dim, vocab_size=527):
        super(EventEncoder, self).__init__()
        self.label_embedding = nn.Parameter(
            torch.randn((vocab_size, emb_dim)), requires_grad=True)
        
    def forward(self, word_idxs):
        indices = word_idxs / word_idxs.sum(dim=1, keepdim=True)
        embeddings = indices @ self.label_embedding
        return embeddings


class EventCondTransformerModel(TransformerModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.EventTransformerDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
        self.label_encoder = EventEncoder(decoder.emb_dim, 527)
        self.train_forward_keys += ["events"]
        self.inference_forward_keys += ["events"]

    # def seq_forward(self, input_dict):
        # cap = input_dict["cap"]
        # cap_padding_mask = (cap == self.pad_idx).to(cap.device)
        # cap_padding_mask = cap_padding_mask[:, :-1]
        # output = self.decoder(
            # {
                # "word": cap[:, :-1],
                # "attn_emb": input_dict["attn_emb"],
                # "attn_emb_len": input_dict["attn_emb_len"],
                # "cap_padding_mask": cap_padding_mask
            # }
        # )
        # return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["events"] = self.label_encoder(input_dict["events"])
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        if t == 0:
            output_i["events"] = repeat_tensor(self.label_encoder(input_dict["events"])[i], beam_size)
        decoder_input["events"] = output_i["events"]
        return decoder_input


class KeywordCondTransformerModel(TransformerModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.KeywordProbTransformerDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
        self.train_forward_keys += ["keyword"]
        self.inference_forward_keys += ["keyword"]

    def seq_forward(self, input_dict):
        cap = input_dict["cap"]
        cap_padding_mask = (cap == self.pad_idx).to(cap.device)
        cap_padding_mask = cap_padding_mask[:, :-1]
        keyword = input_dict["keyword"]
        output = self.decoder(
            {
                "word": cap[:, :-1],
                "attn_emb": input_dict["attn_emb"],
                "attn_emb_len": input_dict["attn_emb_len"],
                "keyword": keyword,
                "cap_padding_mask": cap_padding_mask
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["keyword"] = input_dict["keyword"]
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        if t == 0:
            output_i["keyword"] = repeat_tensor(input_dict["keyword"][i],
                                                 beam_size)
        decoder_input["keyword"] = output_i["keyword"]
        return decoder_input

