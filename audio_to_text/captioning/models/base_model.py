# -*- coding: utf-8 -*-

from typing import Dict

import torch
import torch.nn as nn

from .utils import mean_with_lens, repeat_tensor


class CaptionModel(nn.Module):
    """
    Encoder-decoder captioning model.
    """

    pad_idx = 0
    start_idx = 1
    end_idx = 2
    max_length = 20

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = decoder.vocab_size
        self.train_forward_keys = ["cap", "cap_len", "ss_ratio"]
        self.inference_forward_keys = ["sample_method", "max_length", "temp"]
        freeze_encoder = kwargs.get("freeze_encoder", False)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.check_decoder_compatibility()

    def check_decoder_compatibility(self):
        compatible_decoders = [x.__class__.__name__ for x in self.compatible_decoders]
        assert isinstance(self.decoder, self.compatible_decoders), \
            f"{self.decoder.__class__.__name__} is incompatible with " \
            f"{self.__class__.__name__}, please use decoder in {compatible_decoders} "

    @classmethod
    def set_index(cls, start_idx, end_idx):
        cls.start_idx = start_idx
        cls.end_idx = end_idx

    def forward(self, input_dict: Dict):
        """
        input_dict: {
            (required)
            mode: train/inference,
            spec,
            spec_len,
            fc,
            attn,
            attn_len,
            [sample_method: greedy],
            [temp: 1.0] (in case of no teacher forcing)

            (optional, mode=train)
            cap,
            cap_len,
            ss_ratio,

            (optional, mode=inference)
            sample_method: greedy/beam,
            max_length,
            temp,
            beam_size (optional, sample_method=beam),
            n_best (optional, sample_method=beam),
        }
        """
        # encoder_input_keys = ["spec", "spec_len", "fc", "attn", "attn_len"]
        # encoder_input = { key: input_dict[key] for key in encoder_input_keys }
        encoder_output_dict = self.encoder(input_dict)
        if input_dict["mode"] == "train":
            forward_dict = {
                "mode": "train", "sample_method": "greedy", "temp": 1.0
            }
            for key in self.train_forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            for key in self.inference_forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]

            if forward_dict["sample_method"] == "beam":
                forward_dict["beam_size"] = input_dict.get("beam_size", 3)
                forward_dict["n_best"] = input_dict.get("n_best", False)
                forward_dict["n_best_size"] = input_dict.get("n_best_size", forward_dict["beam_size"])
            elif forward_dict["sample_method"] == "dbs":
                forward_dict["beam_size"] = input_dict.get("beam_size", 6)
                forward_dict["group_size"] = input_dict.get("group_size", 3)
                forward_dict["diversity_lambda"] = input_dict.get("diversity_lambda", 0.5)
                forward_dict["group_nbest"] = input_dict.get("group_nbest", True)

            forward_dict.update(encoder_output_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")

        return output

    def prepare_output(self, input_dict):
        output = {}
        batch_size = input_dict["fc_emb"].size(0)
        if input_dict["mode"] == "train":
            max_length = input_dict["cap"].size(1) - 1
        elif input_dict["mode"] == "inference":
            max_length = input_dict["max_length"]
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        device = input_dict["fc_emb"].device
        output["seq"] = torch.full((batch_size, max_length), self.end_idx,
                                   dtype=torch.long)
        output["logit"] = torch.empty(batch_size, max_length,
                                      self.vocab_size).to(device)
        output["sampled_logprob"] = torch.zeros(batch_size, max_length)
        output["embed"] = torch.empty(batch_size, max_length,
                                      self.decoder.d_model).to(device)
        return output

    def train_forward(self, input_dict):
        if input_dict["ss_ratio"] != 1: # scheduled sampling training
            input_dict["mode"] = "train"
            return self.stepwise_forward(input_dict)
        output = self.seq_forward(input_dict)
        self.train_process(output, input_dict)
        return output

    def seq_forward(self, input_dict):
        raise NotImplementedError

    def train_process(self, output, input_dict):
        pass

    def inference_forward(self, input_dict):
        if input_dict["sample_method"] == "beam":
            return self.beam_search(input_dict)
        elif input_dict["sample_method"] == "dbs":
            return self.diverse_beam_search(input_dict)
        return self.stepwise_forward(input_dict)

    def stepwise_forward(self, input_dict):
        """Step-by-step decoding"""
        output = self.prepare_output(input_dict)
        max_length = output["seq"].size(1)
        # start sampling
        for t in range(max_length):
            input_dict["t"] = t
            self.decode_step(input_dict, output)
            if input_dict["mode"] == "inference": # decide whether to stop when sampling
                unfinished_t = output["seq"][:, t] != self.end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                output["seq"][:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break
        self.stepwise_process(output)
        return output

    def decode_step(self, input_dict, output):
        """Decoding operation of timestep t"""
        decoder_input = self.prepare_decoder_input(input_dict, output)
        # feed to the decoder to get logit
        output_t = self.decoder(decoder_input)
        logit_t = output_t["logit"]
        # assert logit_t.ndim == 3
        if logit_t.size(1) == 1:
            logit_t = logit_t.squeeze(1)
            embed_t = output_t["embed"].squeeze(1)
        elif logit_t.size(1) > 1:
            logit_t = logit_t[:, -1, :]
            embed_t = output_t["embed"][:, -1, :]
        else:
            raise Exception("no logit output")
        # sample the next input word and get the corresponding logit
        sampled = self.sample_next_word(logit_t,
                                        method=input_dict["sample_method"],
                                        temp=input_dict["temp"])

        output_t.update(sampled)
        output_t["t"] = input_dict["t"]
        output_t["logit"] = logit_t
        output_t["embed"] = embed_t
        self.stepwise_process_step(output, output_t)

    def prepare_decoder_input(self, input_dict, output):
        """Prepare the inp ut dict for the decoder"""
        raise NotImplementedError
    
    def stepwise_process_step(self, output, output_t):
        """Postprocessing (save output values) after each timestep t"""
        t = output_t["t"]
        output["logit"][:, t, :] = output_t["logit"]
        output["seq"][:, t] = output_t["word"]
        output["sampled_logprob"][:, t] = output_t["probs"]
        output["embed"][:, t, :] = output_t["embed"]

    def stepwise_process(self, output):
        """Postprocessing after the whole step-by-step autoregressive decoding"""
        pass

    def sample_next_word(self, logit, method, temp):
        """Sample the next word, given probs output by the decoder"""
        logprob = torch.log_softmax(logit, dim=1)
        if method == "greedy":
            sampled_logprob, word = torch.max(logprob.detach(), 1)
        elif method == "gumbel":
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).to(logprob.device)
                return -torch.log(-torch.log(U + eps) + eps)
            def gumbel_softmax_sample(logit, temperature):
                y = logit + sample_gumbel(logit.size())
                return torch.log_softmax(y / temperature, dim=-1)
            _logprob = gumbel_softmax_sample(logprob, temp)
            _, word = torch.max(_logprob.data, 1)
            sampled_logprob = logprob.gather(1, word.unsqueeze(-1))
        else:
            logprob = logprob / temp
            if method.startswith("top"):
                top_num = float(method[3:])
                if 0 < top_num < 1: # top-p sampling
                    probs = torch.softmax(logit, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
                    sorted_probs = sorted_probs * mask.to(sorted_probs)
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprob.scatter_(1, sorted_indices, sorted_probs.log())
                else: # top-k sampling
                    k = int(top_num)
                    tmp = torch.empty_like(logprob).fill_(float('-inf'))
                    topk, indices = torch.topk(logprob, k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprob = tmp
            word = torch.distributions.Categorical(logits=logprob.detach()).sample()
            sampled_logprob = logprob.gather(1, word.unsqueeze(-1)).squeeze(1)
        word = word.detach().long()
        # sampled_logprob: [N,], word: [N,]
        return {"word": word, "probs": sampled_logprob}

    def beam_search(self, input_dict):
        output = self.prepare_output(input_dict)
        max_length = input_dict["max_length"]
        beam_size = input_dict["beam_size"]
        if input_dict["n_best"]:
            n_best_size = input_dict["n_best_size"]
            batch_size, max_length = output["seq"].size()
            output["seq"] = torch.full((batch_size, n_best_size, max_length),
                                        self.end_idx, dtype=torch.long)
            
        temp = input_dict["temp"]
        # instance by instance beam seach
        for i in range(output["seq"].size(0)):
            output_i = self.prepare_beamsearch_output(input_dict)
            input_dict["sample_idx"] = i
            for t in range(max_length):
                input_dict["t"] = t
                output_t = self.beamsearch_step(input_dict, output_i)
                #######################################
                # merge with previous beam and select the current max prob beam
                #######################################
                logit_t = output_t["logit"]
                if logit_t.size(1) == 1:
                    logit_t = logit_t.squeeze(1)
                elif logit_t.size(1) > 1:
                    logit_t = logit_t[:, -1, :]
                else:
                    raise Exception("no logit output")
                logprob_t = torch.log_softmax(logit_t, dim=1)
                logprob_t = torch.log_softmax(logprob_t / temp, dim=1)
                logprob_t = output_i["topk_logprob"].unsqueeze(1) + logprob_t
                if t == 0: # for the first step, all k seq will have the same probs
                    topk_logprob, topk_words = logprob_t[0].topk(
                        beam_size, 0, True, True)
                else: # unroll and find top logprob, and their unrolled indices
                    topk_logprob, topk_words = logprob_t.view(-1).topk(
                        beam_size, 0, True, True)
                topk_words = topk_words.cpu()
                output_i["topk_logprob"] = topk_logprob
                # output_i["prev_words_beam"] = topk_words // self.vocab_size  # [beam_size,]
                output_i["prev_words_beam"] = torch.div(topk_words, self.vocab_size,
                                                        rounding_mode='trunc')
                output_i["next_word"] = topk_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seq"] = output_i["next_word"].unsqueeze(1)
                else:
                    output_i["seq"] = torch.cat([
                        output_i["seq"][output_i["prev_words_beam"]],
                        output_i["next_word"].unsqueeze(1)], dim=1)

                # add finished beams to results
                is_end = output_i["next_word"] == self.end_idx
                if t == max_length - 1:
                    is_end.fill_(1)
                
                for beam_idx in range(beam_size):
                    if is_end[beam_idx]:
                        final_beam = {
                            "seq": output_i["seq"][beam_idx].clone(),
                            "score": output_i["topk_logprob"][beam_idx].item()
                        }
                        final_beam["score"] = final_beam["score"] / (t + 1)
                        output_i["done_beams"].append(final_beam)
                output_i["topk_logprob"][is_end] -= 1000

                self.beamsearch_process_step(output_i, output_t)

            self.beamsearch_process(output, output_i, input_dict)
        return output

    def prepare_beamsearch_output(self, input_dict):
        beam_size = input_dict["beam_size"]
        device = input_dict["fc_emb"].device
        output = {
            "topk_logprob": torch.zeros(beam_size).to(device),
            "seq": None,
            "prev_words_beam": None,
            "next_word": None,
            "done_beams": [],
        }
        return output

    def beamsearch_step(self, input_dict, output_i):
        decoder_input = self.prepare_beamsearch_decoder_input(input_dict, output_i)
        output_t = self.decoder(decoder_input)
        output_t["t"] = input_dict["t"]
        return output_t

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        raise NotImplementedError
            
    def beamsearch_process_step(self, output_i, output_t):
        pass

    def beamsearch_process(self, output, output_i, input_dict):
        i = input_dict["sample_idx"]
        done_beams = sorted(output_i["done_beams"], key=lambda x: -x["score"])
        if input_dict["n_best"]:
            done_beams = done_beams[:input_dict["n_best_size"]]
            for out_idx, done_beam in enumerate(done_beams):
                seq = done_beam["seq"]
                output["seq"][i][out_idx, :len(seq)] = seq
        else:
            seq = done_beams[0]["seq"]
            output["seq"][i][:len(seq)] = seq
    
    def diverse_beam_search(self, input_dict):
        
        def add_diversity(seq_table, logprob, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprob = logprob.clone()

            if divm > 0:
                change = torch.zeros(logprob.size(-1))
                for prev_choice in range(divm):
                    prev_decisions = seq_table[prev_choice][..., local_time]
                    for prev_labels in range(bdash):
                        change.scatter_add_(0, prev_decisions[prev_labels], change.new_ones(1))

                change = change.to(logprob.device)
                logprob = logprob - repeat_tensor(change, bdash) * diversity_lambda

            return logprob, unaug_logprob

        output = self.prepare_output(input_dict)
        group_size = input_dict["group_size"]
        batch_size = output["seq"].size(0)
        beam_size = input_dict["beam_size"]
        bdash = beam_size // group_size
        input_dict["bdash"] = bdash
        diversity_lambda = input_dict["diversity_lambda"]
        device = input_dict["fc_emb"].device
        max_length = input_dict["max_length"]
        temp = input_dict["temp"]
        group_nbest = input_dict["group_nbest"]
        batch_size, max_length = output["seq"].size()
        if group_nbest:
            output["seq"] = torch.full((batch_size, beam_size, max_length),
                                        self.end_idx, dtype=torch.long)
        else:
            output["seq"] = torch.full((batch_size, group_size, max_length),
                                        self.end_idx, dtype=torch.long)


        for i in range(batch_size):
            input_dict["sample_idx"] = i
            seq_table = [torch.LongTensor(bdash, 0) for _ in range(group_size)] # group_size x [bdash, 0]
            logprob_table = [torch.zeros(bdash).to(device) for _ in range(group_size)]
            done_beams_table = [[] for _ in range(group_size)]

            output_i = {
                "prev_words_beam": [None for _ in range(group_size)],
                "next_word": [None for _ in range(group_size)],
                "state": [None for _ in range(group_size)]
            }

            for t in range(max_length + group_size - 1):
                input_dict["t"] = t
                for divm in range(group_size):
                    input_dict["divm"] = divm
                    if t >= divm and t <= max_length + divm - 1:
                        local_time = t - divm
                        decoder_input = self.prepare_dbs_decoder_input(input_dict, output_i)
                        output_t = self.decoder(decoder_input)
                        output_t["divm"] = divm
                        logit_t = output_t["logit"]
                        if logit_t.size(1) == 1:
                            logit_t = logit_t.squeeze(1)
                        elif logit_t.size(1) > 1:
                            logit_t = logit_t[:, -1, :]
                        else:
                            raise Exception("no logit output")
                        logprob_t = torch.log_softmax(logit_t, dim=1)
                        logprob_t = torch.log_softmax(logprob_t / temp, dim=1)
                        logprob_t, unaug_logprob_t = add_diversity(seq_table, logprob_t, t, divm, diversity_lambda, bdash)
                        logprob_t = logprob_table[divm].unsqueeze(-1) + logprob_t
                        if local_time == 0: # for the first step, all k seq will have the same probs
                            topk_logprob, topk_words = logprob_t[0].topk(
                                bdash, 0, True, True)
                        else: # unroll and find top logprob, and their unrolled indices
                            topk_logprob, topk_words = logprob_t.view(-1).topk(
                                bdash, 0, True, True)
                        topk_words = topk_words.cpu()
                        logprob_table[divm] = topk_logprob
                        output_i["prev_words_beam"][divm] = topk_words // self.vocab_size  # [bdash,]
                        output_i["next_word"][divm] = topk_words % self.vocab_size  # [bdash,]
                        if local_time > 0:
                            seq_table[divm] = seq_table[divm][output_i["prev_words_beam"][divm]]
                        seq_table[divm] = torch.cat([
                            seq_table[divm],
                            output_i["next_word"][divm].unsqueeze(-1)], -1)

                        is_end = seq_table[divm][:, t-divm] == self.end_idx
                        assert seq_table[divm].shape[-1] == t - divm + 1
                        if t == max_length + divm - 1:
                            is_end.fill_(1)
                        for beam_idx in range(bdash):
                            if is_end[beam_idx]:
                                final_beam = {
                                    "seq": seq_table[divm][beam_idx].clone(),
                                    "score": logprob_table[divm][beam_idx].item()
                                }
                                final_beam["score"] = final_beam["score"] / (t - divm + 1)
                                done_beams_table[divm].append(final_beam)
                        logprob_table[divm][is_end] -= 1000
                        self.dbs_process_step(output_i, output_t)
            done_beams_table = [sorted(done_beams_table[divm], key=lambda x: -x["score"])[:bdash] for divm in range(group_size)]
            if group_nbest:
                done_beams = sum(done_beams_table, [])
            else:
                done_beams = [group_beam[0] for group_beam in done_beams_table]
            for _, done_beam in enumerate(done_beams):
                output["seq"][i, _, :len(done_beam["seq"])] = done_beam["seq"]

        return output
            
    def prepare_dbs_decoder_input(self, input_dict, output_i):
        raise NotImplementedError

    def dbs_process_step(self, output_i, output_t):
        pass


class CaptionSequenceModel(nn.Module):

    def __init__(self, model, seq_output_size):
        super().__init__()
        self.model = model
        if model.decoder.d_model != seq_output_size:
            self.output_transform = nn.Linear(model.decoder.d_model, seq_output_size)
        else:
            self.output_transform = lambda x: x

    def forward(self, input_dict):
        output = self.model(input_dict)

        if input_dict["mode"] == "train":
            lens = input_dict["cap_len"] - 1
            # seq_outputs: [N, d_model]
        elif input_dict["mode"] == "inference":
            if "sample_method" in input_dict and input_dict["sample_method"] == "beam":
                return output
            seq = output["seq"]
            lens = torch.where(seq == self.model.end_idx, torch.zeros_like(seq), torch.ones_like(seq)).sum(dim=1)
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        seq_output = mean_with_lens(output["embed"], lens)
        seq_output = self.output_transform(seq_output)
        output["seq_output"] = seq_output
        return output

