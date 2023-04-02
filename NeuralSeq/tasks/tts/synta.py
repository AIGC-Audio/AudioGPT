import os
import torch
import torch.nn.functional as F
from torch import nn

from modules.tts.syntaspeech.syntaspeech import SyntaSpeech
from tasks.tts.ps_adv import PortaSpeechAdvTask
from utils.hparams import hparams


class SyntaSpeechTask(PortaSpeechAdvTask):
    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = SyntaSpeech(ph_dict_size, word_dict_size, hparams)
    
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
        self.dp_params = [p for k, p in self.model.named_parameters() if (('dur_predictor' in k) and p.requires_grad)]
        self.gen_params_except_dp = [p for k, p in self.model.named_parameters() if (('dur_predictor' not in k) and p.requires_grad)]        
        self.bert_params = [p for k, p in self.model.named_parameters() if (('bert' in k) and p.requires_grad)]
        self.gen_params_except_bert_and_dp = [p for k, p in self.model.named_parameters() if ('dur_predictor' not in k) and ('bert' not in k) and p.requires_grad ]

        self.use_bert = True if len(self.bert_params) > 0 else False
    
    