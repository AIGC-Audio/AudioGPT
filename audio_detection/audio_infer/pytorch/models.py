import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from audio_infer.pytorch.pytorch_utils import do_mixup, interpolate, pad_framewise_output
import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from audio_infer.pytorch.pytorch_utils import do_mixup
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import warnings
from functools import partial
#from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
os.environ['TORCH_HOME'] = '../pretrained_models'
from copy import deepcopy
from timm.models.helpers import load_pretrained
from torch.cuda.amp import autocast
from collections import OrderedDict
import io
import re
from mmcv.runner import _load_checkpoint, load_state_dict
import mmcv.runner
import copy
import random
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location, logger)
    new_proj = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=1).unsqueeze(1))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    state_dict = OrderedDict({k.replace('backbone.',''):v for k,v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)




class TimeShift(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            shift = torch.empty(1).normal_(self.mean, self.std).int().item()
            x = torch.roll(x, shift, dims=2)
        return x

class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)

class PVT(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(PVT, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.time_shift = TimeShift(0, 10)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
 
        self.bn0 = nn.BatchNorm2d(64)
        self.pvt_transformer = PyramidVisionTransformerV2(tdim=1001,
                                fdim=64,
                                patch_size=7,
                                stride=4,
                                in_chans=1,
                                num_classes=classes_num,
                                embed_dims=[64, 128, 320, 512],
                                depths=[3, 4, 6, 3],
                                num_heads=[1, 2, 5, 8],
                                mlp_ratios=[8, 8, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                sr_ratios=[8, 4, 2, 1],
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                num_stages=4,
                                #pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
                                )
        #self.temp_pool = LinearSoftPool()  
        self.avgpool = nn.AdaptiveAvgPool1d(1)    
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 32

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.time_shift(x)
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        #print(x.shape)   #torch.Size([10, 1, 1001, 64])
        x = self.pvt_transformer(x)
        #print(x.shape)   #torch.Size([10, 800, 128])
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2).contiguous()
        framewise_output = torch.sigmoid(self.fc_audioset(x))
        #clipwise_output = torch.mean(framewise_output, dim=1)
        #clipwise_output = self.temp_pool(x, framewise_output).clamp(1e-7, 1.).squeeze(1)
        x = framewise_output.transpose(1, 2).contiguous()
        x = self.avgpool(x)
        clipwise_output = torch.flatten(x, 1)
        #print(framewise_output.shape)    #torch.Size([10, 100, 17])
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = framewise_output[:,:1000,:]
        #framewise_output = pad_framewise_output(framewise_output, frames_num)
        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}
            
        return output_dict

class PVT2(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(PVT2, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.time_shift = TimeShift(0, 10)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
 
        self.bn0 = nn.BatchNorm2d(64)
        self.pvt_transformer = PyramidVisionTransformerV2(tdim=1001,
                                fdim=64,
                                patch_size=7,
                                stride=4,
                                in_chans=1,
                                num_classes=classes_num,
                                embed_dims=[64, 128, 320, 512],
                                depths=[3, 4, 6, 3],
                                num_heads=[1, 2, 5, 8],
                                mlp_ratios=[8, 8, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                sr_ratios=[8, 4, 2, 1],
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                num_stages=4,
                                pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
                                )
        #self.temp_pool = LinearSoftPool()      
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 32

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            #x = self.time_shift(x)
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        #print(x.shape)   #torch.Size([10, 1, 1001, 64])
        x = self.pvt_transformer(x)
        #print(x.shape)   #torch.Size([10, 800, 128])
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2).contiguous()
        framewise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output = torch.mean(framewise_output, dim=1)
        #clipwise_output = self.temp_pool(x, framewise_output).clamp(1e-7, 1.).squeeze(1)
        #print(framewise_output.shape)    #torch.Size([10, 100, 17])
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = framewise_output[:,:1000,:]
        #framewise_output = pad_framewise_output(framewise_output, frames_num)
        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}
            
        return output_dict

class PVT_2layer(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(PVT_2layer, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.time_shift = TimeShift(0, 10)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
 
        self.bn0 = nn.BatchNorm2d(64)
        self.pvt_transformer = PyramidVisionTransformerV2(tdim=1001,
                                fdim=64,
                                patch_size=7,
                                stride=4,
                                in_chans=1,
                                num_classes=classes_num,
                                embed_dims=[64, 128],
                                depths=[3, 4],
                                num_heads=[1, 2],
                                mlp_ratios=[8, 8],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                sr_ratios=[8, 4],
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                num_stages=2,
                                pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
                                )
        #self.temp_pool = LinearSoftPool()  
        self.avgpool = nn.AdaptiveAvgPool1d(1)    
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.time_shift(x)
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        #print(x.shape)   #torch.Size([10, 1, 1001, 64])
        x = self.pvt_transformer(x)
        #print(x.shape)   #torch.Size([10, 800, 128])
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2).contiguous()
        framewise_output = torch.sigmoid(self.fc_audioset(x))
        #clipwise_output = torch.mean(framewise_output, dim=1)
        #clipwise_output = self.temp_pool(x, framewise_output).clamp(1e-7, 1.).squeeze(1)
        x = framewise_output.transpose(1, 2).contiguous()
        x = self.avgpool(x)
        clipwise_output = torch.flatten(x, 1)
        #print(framewise_output.shape)    #torch.Size([10, 100, 17])
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = framewise_output[:,:1000,:]
        #framewise_output = pad_framewise_output(framewise_output, frames_num)
        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}
            
        return output_dict

class PVT_lr(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(PVT_lr, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.time_shift = TimeShift(0, 10)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
 
        self.bn0 = nn.BatchNorm2d(64)
        self.pvt_transformer = PyramidVisionTransformerV2(tdim=1001,
                                fdim=64,
                                patch_size=7,
                                stride=4,
                                in_chans=1,
                                num_classes=classes_num,
                                embed_dims=[64, 128, 320, 512],
                                depths=[3, 4, 6, 3],
                                num_heads=[1, 2, 5, 8],
                                mlp_ratios=[8, 8, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                sr_ratios=[8, 4, 2, 1],
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                num_stages=4,
                                pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
                                )
        self.temp_pool = LinearSoftPool()      
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 32

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.time_shift(x)
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        #print(x.shape)   #torch.Size([10, 1, 1001, 64])
        x = self.pvt_transformer(x)
        #print(x.shape)   #torch.Size([10, 800, 128])
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2).contiguous()
        framewise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output = self.temp_pool(x, framewise_output).clamp(1e-7, 1.).squeeze(1)
        #print(framewise_output.shape)    #torch.Size([10, 100, 17])
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = framewise_output[:,:1000,:]
        #framewise_output = pad_framewise_output(framewise_output, frames_num)
        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}
            
        return output_dict


class PVT_nopretrain(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(PVT_nopretrain, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.time_shift = TimeShift(0, 10)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
 
        self.bn0 = nn.BatchNorm2d(64)
        self.pvt_transformer = PyramidVisionTransformerV2(tdim=1001,
                                fdim=64,
                                patch_size=7,
                                stride=4,
                                in_chans=1,
                                num_classes=classes_num,
                                embed_dims=[64, 128, 320, 512],
                                depths=[3, 4, 6, 3],
                                num_heads=[1, 2, 5, 8],
                                mlp_ratios=[8, 8, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                sr_ratios=[8, 4, 2, 1],
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                num_stages=4,
                                #pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
                                )
        self.temp_pool = LinearSoftPool()      
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 32

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.time_shift(x)
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        #print(x.shape)   #torch.Size([10, 1, 1001, 64])
        x = self.pvt_transformer(x)
        #print(x.shape)   #torch.Size([10, 800, 128])
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2).contiguous()
        framewise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output = self.temp_pool(x, framewise_output).clamp(1e-7, 1.).squeeze(1)
        #print(framewise_output.shape)    #torch.Size([10, 100, 17])
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        framewise_output = framewise_output[:,:1000,:]
        #framewise_output = pad_framewise_output(framewise_output, frames_num)
        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}
            
        return output_dict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        #self.norm3 = norm_layer(dim)
        #self.token_mixer = Pooling(pool_size=3)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, tdim, fdim, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (tdim, fdim)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 3, patch_size[1] // 3))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, tdim=1001, fdim=64, patch_size=16, stride=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=2, linear=False, pretrained=None):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(tdim=tdim if i == 0 else tdim // (2 ** (i + 1)),
                                            fdim=fdim if i == 0 else tdim // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=stride if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
 
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        #self.n = nn.Linear(125, 250, bias=True)
        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            #print(x.shape)
            for blk in block:
                x = blk(x, H, W)
            #print(x.shape)
            x = norm(x)
            #if i != self.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print(x.shape)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict
