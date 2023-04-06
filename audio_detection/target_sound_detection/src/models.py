# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
from itertools import zip_longest
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import math
from sklearn.cluster import KMeans
import os
import time
from functools import partial
# import timm
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import warnings
from functools import partial
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
# from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint
# from mmcv.runner import _load_checkpoint, load_state_dict
# import mmcv.runner
import copy
from collections import OrderedDict
import io
import re
DEBUG=0
event_labels = ['Alarm', 'Alarm_clock', 'Animal', 'Applause', 'Arrow', 'Artillery_fire', 
                'Babbling', 'Baby_laughter', 'Bark', 'Basketball_bounce', 'Battle_cry', 
                'Bell', 'Bird', 'Bleat', 'Bouncing', 'Breathing', 'Buzz', 'Camera', 
                'Cap_gun', 'Car', 'Car_alarm', 'Cat', 'Caw', 'Cheering', 'Child_singing', 
                'Choir', 'Chop', 'Chopping_(food)', 'Clapping', 'Clickety-clack', 'Clicking', 
                'Clip-clop', 'Cluck', 'Coin_(dropping)', 'Computer_keyboard', 'Conversation', 
                'Coo', 'Cough', 'Cowbell', 'Creak', 'Cricket', 'Croak', 'Crow', 'Crowd', 'DTMF', 
                'Dog', 'Door', 'Drill', 'Drip', 'Engine', 'Engine_starting', 'Explosion', 'Fart', 
                'Female_singing', 'Filing_(rasp)', 'Finger_snapping', 'Fire', 'Fire_alarm', 'Firecracker', 
                'Fireworks', 'Frog', 'Gasp', 'Gears', 'Giggle', 'Glass', 'Glass_shatter', 'Gobble', 'Groan', 
                'Growling', 'Hammer', 'Hands', 'Hiccup', 'Honk', 'Hoot', 'Howl', 'Human_sounds', 'Human_voice', 
                'Insect', 'Laughter', 'Liquid', 'Machine_gun', 'Male_singing', 'Mechanisms', 'Meow', 'Moo', 
                'Motorcycle', 'Mouse', 'Music', 'Oink', 'Owl', 'Pant', 'Pant_(dog)', 'Patter', 'Pig', 'Plop',
                'Pour', 'Power_tool', 'Purr', 'Quack', 'Radio', 'Rain_on_surface', 'Rapping', 'Rattle', 
                'Reversing_beeps', 'Ringtone', 'Roar', 'Run', 'Rustle', 'Scissors', 'Scrape', 'Scratch', 
                'Screaming', 'Sewing_machine', 'Shout', 'Shuffle', 'Shuffling_cards', 'Singing', 
                'Single-lens_reflex_camera', 'Siren', 'Skateboard', 'Sniff', 'Snoring', 'Speech', 
                'Speech_synthesizer', 'Spray', 'Squeak', 'Squeal', 'Steam', 'Stir', 'Surface_contact', 
                'Tap', 'Tap_dance', 'Telephone_bell_ringing', 'Television', 'Tick', 'Tick-tock', 'Tools', 
                'Train', 'Train_horn', 'Train_wheels_squealing', 'Truck', 'Turkey', 'Typewriter', 'Typing', 
                'Vehicle', 'Video_game_sound', 'Water', 'Whimper_(dog)', 'Whip', 'Whispering', 'Whistle', 
                'Whistling', 'Whoop', 'Wind', 'Writing', 'Yip', 'and_pans', 'bird_song', 'bleep', 'clink', 
                'cock-a-doodle-doo', 'crinkling', 'dove', 'dribble', 'eructation', 'faucet', 'flapping_wings', 
                'footsteps', 'gunfire', 'heartbeat', 'infant_cry', 'kid_speaking', 'man_speaking', 'mastication', 
                'mice', 'river', 'rooster', 'silverware', 'skidding', 'smack', 'sobbing', 'speedboat', 'splatter',
                'surf', 'thud', 'thwack', 'toot', 'truck_horn', 'tweet', 'vroom', 'waterfowl', 'woman_speaking']
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
    '''
    new_proj = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=1).unsqueeze(1))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=2).unsqueeze(2).repeat(1,1,3,1))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    new_proj.weight = torch.nn.Parameter(torch.sum(checkpoint['patch_embed1.proj.weight'], dim=3).unsqueeze(3).repeat(1,1,1,3))
    checkpoint['patch_embed1.proj.weight'] = new_proj.weight
    '''
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

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
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

class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]


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
        return (time_decision**2).sum(self.pooldim) / (time_decision.sum(
            self.pooldim)+1e-7)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvBlock_GLU(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(3,3)):
        super(ConvBlock_GLU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              padding=(1, 1), bias=False)                         
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = self.bn1(self.conv1(x))
        cnn1 = self.sigmoid(x[:, :x.shape[1]//2, :, :])
        cnn2 = x[:,x.shape[1]//2:,:,:]
        x = cnn1*cnn2
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        elif pool_type == 'None':
            pass
        elif pool_type == 'LP':
            pass
            #nn.LPPool2d(4, pool_size)
        else:
            raise Exception('Incorrect argument!')
        return x

class Mul_scale_GLU(nn.Module):
    def __init__(self):
        super(Mul_scale_GLU,self).__init__()
        self.conv_block1_1 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(1,1)) # 1*1
        self.conv_block1_2 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(3,3)) # 3*3
        self.conv_block1_3 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(5,5)) # 5*5
        self.conv_block2 = ConvBlock_GLU(in_channels=96, out_channels=128*2)
        # self.conv_block3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock_GLU(in_channels=128, out_channels=128*2)
        self.conv_block4 = ConvBlock_GLU(in_channels=128, out_channels=256*2)
        self.conv_block5 = ConvBlock_GLU(in_channels=256, out_channels=256*2)
        self.conv_block6 = ConvBlock_GLU(in_channels=256, out_channels=512*2)
        self.conv_block7 = ConvBlock_GLU(in_channels=512, out_channels=512*2)
        self.padding = nn.ReplicationPad2d((0,1,0,1))

    def forward(self, input, fi=None):
        """
        Input: (batch_size, data_length)"""
        x1 = self.conv_block1_1(input, pool_size=(2, 2), pool_type='avg')
        x1 = x1[:,:,:500,:32]
        #print('x1 ',x1.shape)
        x2 = self.conv_block1_2(input,pool_size=(2,2),pool_type='avg')
        #print('x2 ',x2.shape)
        x3 = self.conv_block1_3(input,pool_size=(2,2),pool_type='avg')
        x3 = self.padding(x3)
        #print('x3 ',x3.shape)
        # assert 1==2
        x = torch.cat([x1,x2],dim=1)
        x = torch.cat([x,x3],dim=1)
        #print('x ',x.shape)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='None')
        x = self.conv_block3(x,pool_size=(2,2),pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # 
        #print('x2,3 ',x.shape)
        x = self.conv_block4(x, pool_size=(2, 4), pool_type='None')
        x = self.conv_block5(x,pool_size=(2,4),pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        #print('x4,5 ',x.shape)

        x = self.conv_block6(x, pool_size=(1, 4), pool_type='None')
        x = self.conv_block7(x, pool_size=(1, 4), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print('x6,7 ',x.shape)
        # assert 1==2
        return x

class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527):
        
        super(Cnn14, self).__init__()

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

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input_, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        input_ = input_.unsqueeze(1)
        x = self.conv_block1(input_, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.shape)
        # x = torch.mean(x, dim=3)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = self.fc1(x)
        # print(x.shape)
        # assert 1==2
        # (x1,_) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        return x

class Cnn10_fi(nn.Module):
    def __init__(self):  
        super(Cnn10_fi, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # self.fc1 = nn.Linear(512, 512, bias=True)
        # self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        # self.init_weight()
 
    def forward(self, input, fi=None):
        """
        Input: (batch_size, data_length)"""

        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 4), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 4), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class Cnn10_mul_scale(nn.Module):
    def __init__(self,scale=8):  
        super(Cnn10_mul_scale, self).__init__()
        self.conv_block1_1 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(1,1))
        self.conv_block1_2 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(3,3))
        self.conv_block1_3 = ConvBlock_GLU(in_channels=1, out_channels=64,kernel_size=(5,5))
        self.conv_block2 = ConvBlock(in_channels=96, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.scale = scale
        self.padding = nn.ReplicationPad2d((0,1,0,1))
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Input: (batch_size, data_length)"""
        if self.scale == 8:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (2,4)
            pool_size4 = (1,4)
        elif self.scale == 4:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        elif self.scale == 2:
            pool_size1 = (2,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        else:
            pool_size1 = (1,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        # print('input ',input.shape)
        x1 = self.conv_block1_1(input, pool_size=pool_size1, pool_type='avg')
        x1 = x1[:,:,:500,:32]
        #print('x1 ',x1.shape)
        x2 = self.conv_block1_2(input, pool_size=pool_size1, pool_type='avg')
        #print('x2 ',x2.shape)
        x3 = self.conv_block1_3(input, pool_size=pool_size1, pool_type='avg')
        x3 = self.padding(x3)
        #print('x3 ',x3.shape)
        # assert 1==2
        m_i = min(x3.shape[2],min(x1.shape[2],x2.shape[2]))
        #print('m_i ', m_i)
        x = torch.cat([x1[:,:,:m_i,:],x2[:,:, :m_i,:],x3[:,:, :m_i,:]],dim=1)
        # x = torch.cat([x,x3],dim=1)

        # x = self.conv_block1(input, pool_size=pool_size1, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=pool_size2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=pool_size3, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=pool_size4, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x


class Cnn10(nn.Module):
    def __init__(self,scale=8):  
        super(Cnn10, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.scale = scale
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Input: (batch_size, data_length)"""
        if self.scale == 8:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (2,4)
            pool_size4 = (1,4)
        elif self.scale == 4:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        elif self.scale == 2:
            pool_size1 = (2,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        else:
            pool_size1 = (1,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        x = self.conv_block1(input, pool_size=pool_size1, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=pool_size2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=pool_size3, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=pool_size4, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)

class ResPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.linPool = LinearSoftPool(pooldim=1)

class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class ExtAttentionPool(nn.Module):
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.attention = nn.Linear(inputdim, outputdim)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.activ = nn.Softmax(dim=self.pooldim)

    def forward(self, logits, decision):
        # Logits of shape (B, T, D), decision of shape (B, T, C)
        w_x = self.activ(self.attention(logits) / self.outputdim)
        h = (logits.permute(0, 2, 1).contiguous().unsqueeze(-2) *
             w_x.unsqueeze(-1)).flatten(-2).contiguous()
        return torch.sum(h, self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)

class AudioCNN(nn.Module):
    def __init__(self, classes_num):
        super(AudioCNN, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512,128,bias=True)
        self.fc = nn.Linear(128, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        # [128, 801, 168] --> [128,1,801,168]
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # 128,64,400,84
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg') # 128,128,200,42
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # 128,256,100,21
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') # 128,512,50,10
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes) # 128,512,50
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps) 128,512
        x = self.fc1(x) # 128,128
        output = self.fc(x) # 128,10
        return x,output

    def extract(self,input):
        '''Input: (batch_size, times_steps, freq_bins)'''
        x = input[:, None, :, :]
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc1(x) # 128,128
        return x

def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])
class conv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding='VALID', dilation=1):
        super(conv1d, self).__init__()
        if padding == 'VALID':
            dconv_pad = 0
        elif padding == 'SAME':
            dconv_pad = dilation * ((kernel_size - 1) // 2)
        else:
            raise ValueError("Padding Mode Error!")
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=dconv_pad)
        self.act = nn.ReLU()
        self.init_layer(self.conv)

    def init_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out

class Atten_1(nn.Module):
    def __init__(self, input_dim, context=2, dropout_rate=0.2):
        super(Atten_1, self).__init__()
        self._matrix_k = nn.Linear(input_dim, input_dim // 4)
        self._matrix_q = nn.Linear(input_dim, input_dim // 4)
        self.relu = nn.ReLU()
        self.context = context
        self._dropout_layer = nn.Dropout(dropout_rate)
        self.init_layer(self._matrix_k)
        self.init_layer(self._matrix_q)

    def init_layer(self, layer, nonlinearity='leaky_relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def forward(self, input_x):
        k_x = input_x
        k_x = self.relu(self._matrix_k(k_x))
        k_x = self._dropout_layer(k_x)
        # print('k_x ',k_x.shape)
        q_x = input_x[:, self.context, :]
        # print('q_x ',q_x.shape)
        q_x = q_x[:, None, :]
        # print('q_x1 ',q_x.shape)
        q_x = self.relu(self._matrix_q(q_x))
        q_x = self._dropout_layer(q_x)
        # print('q_x2 ',q_x.shape)
        x_ = torch.matmul(k_x, q_x.transpose(-2, -1) / math.sqrt(k_x.size(-1)))
        # print('x_ ',x_.shape)
        x_ = x_.squeeze(2)
        alpha = F.softmax(x_, dim=-1)
        att_ = alpha
        # print('alpha ',alpha)
        alpha = alpha.unsqueeze(2).repeat(1,1,input_x.shape[2])
        # print('alpha ',alpha)
        # alpha = alpha.view(alpha.size(0), alpha.size(1), alpha.size(2), 1)
        out = alpha * input_x
        # print('out ', out.shape)
        # out = out.mean(2)
        out = out.mean(1)
        # print('out ',out.shape)
        # assert 1==2
        #y = alpha * input_x
        #return y, att_
        out = input_x[:, self.context, :] + out
        return out

class Fusion(nn.Module):
    def __init__(self, inputdim, inputdim2, n_fac):
        super().__init__()
        self.fuse_layer1 = conv1d(inputdim, inputdim2*n_fac,1)
        self.fuse_layer2 = conv1d(inputdim2, inputdim2*n_fac,1)
        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac) # 沿着最后一个维度进行pooling

    def forward(self,embedding,mix_embed):
        embedding = embedding.permute(0,2,1)
        fuse1_out = self.fuse_layer1(embedding) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse1_out = fuse1_out.permute(0,2,1)

        mix_embed = mix_embed.permute(0,2,1)
        fuse2_out = self.fuse_layer2(mix_embed) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse2_out = fuse2_out.permute(0,2,1)
        as_embs = torch.mul(fuse1_out, fuse2_out) # 相乘 [2, 501, 2560]
        # (10, 501, 512)
        as_embs = self.avg_pool(as_embs) # [2, 501, 512] 相当于 2560//5
        return as_embs

class CDur_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.fusion = Fusion(128,2)
        self.fc = nn.Linear(256,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding,x)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_big(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 2)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (2, 4)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),)
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_GLU(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Mul_scale_GLU()
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512,1, bidirectional=True, batch_first=True) # previous is 640
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        
    def forward(self, x, embedding,one_hot=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)

        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution,**kwargs):
        super().__init__()
        if time_resolution==125:
            self.features = Cnn10(8)
        elif time_resolution == 250:
            #print('time_resolution ',time_resolution)
            self.features = Cnn10(4)
        elif time_resolution == 500:
            self.features = Cnn10(2)
        else:
            self.features = Cnn10(0)
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.features = Cnn10()
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,one_hot=None):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN_mul_scale(nn.Module):
    def __init__(self, inputdim, outputdim,time_resolution,**kwargs):
        super().__init__()
        if time_resolution==125:
            self.features = Cnn10_mul_scale(8)
        elif time_resolution == 250:
            #print('time_resolution ',time_resolution)
            self.features = Cnn10_mul_scale(4)
        elif time_resolution == 500:
            self.features = Cnn10_mul_scale(2)
        else:
            self.features = Cnn10_mul_scale(0)
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.features = Cnn10()
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,one_hot=None):
        # print('x ',x.shape)
        # assert 1==2
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN_mul_scale_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, time_resolution,**kwargs):
        super().__init__()
        if time_resolution==125:
            self.features = Cnn10_mul_scale(8)
        elif time_resolution == 250:
            #print('time_resolution ',time_resolution)
            self.features = Cnn10_mul_scale(4)
        elif time_resolution == 500:
            self.features = Cnn10_mul_scale(2)
        else:
            self.features = Cnn10_mul_scale(0)
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.features = Cnn10()
        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.fusion = Fusion(128,512,2)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    
    def forward(self, x, embedding,one_hot=None):
        # print('x ',x.shape)
        # assert 1==2
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding, x)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up


class RaDur_fusion(nn.Module):
    def __init__(self, model_config, inputdim, outputdim, time_resolution, **kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN_mul_scale_fusion(inputdim, outputdim, time_resolution)
        self.softmax = nn.Softmax(dim=2)
        #self.temperature = 5
        # if model_config['pre_train']:
        #     self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
        #     self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        
        self.q = nn.Linear(128,128)
        self.k = nn.Linear(128,128)
        self.q_ee = nn.Linear(128, 128)
        self.k_ee = nn.Linear(128, 128)
        self.temperature = 11.3 # sqrt(128)
        self.att_pool = model_config['att_pool']
        self.enhancement = model_config['enhancement'] 
        self.tao = model_config['tao']
        self.top = model_config['top']
        self.bn = nn.BatchNorm1d(128)
        self.EE_fusion = Fusion(128, 128, 4)

    def get_w(self,q,k):
        q = self.q(q)
        k = self.k(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def get_w_ee(self,q,k):
        q = self.q_ee(q)
        k = self.k_ee(k)
        q = q.unsqueeze(1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self, embeddings, mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        # print('top_k ', top_k)
        # top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        # print('index ', idx_topk)
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def sum_with_attention(self, embedding, top_k, selected_embeddings):
        # print('embedding ',embedding)
        # print('selected_embeddings ',selected_embeddings.shape)
        att_1 = self.get_w_ee(embedding, selected_embeddings)
        att_1 = att_1.squeeze(1)
        #print('att_1 ',att_1.shape)
        larger = top_k > self.tao
        # print('larger ',larger)
        top_k = top_k*larger
        # print('top_k ',top_k.shape)
        # print('top_k ',top_k)
        att_1 = att_1*top_k
        #print('att_1 ',att_1.shape)
        # assert 1==2
        att_2 = att_1.unsqueeze(2).repeat(1,1,128)
        Es = selected_embeddings*att_2
        return Es
    
    def orcal_EE(self, x, embedding, label):
        batch, time, dim = x.shape

        mixture_embedding = self.encoder(x) # 8, 125, 128
        mixture_embedding = mixture_embedding.transpose(1,2)
        mixture_embedding = self.bn(mixture_embedding)
        mixture_embedding = mixture_embedding.transpose(1,2)

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        f = self.detection.fusion(embedding_pre, x) # the first stage results
        #f = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        f, _ = self.detection.gru(f) #  x  torch.Size([16, 125, 256])
        f = self.detection.fc(f)
        decision_time = torch.softmax(self.detection.outputlayer(f),dim=2) # x  torch.Size([16, 125, 2])
        
        selected_embeddings, top_k = self.select_topk_embeddings(decision_time[:,:,0], mixture_embedding, self.top)
        
        selected_embeddings = self.sum_with_attention(embedding, top_k, selected_embeddings) # add the weight

        mix_embedding = selected_embeddings.mean(1).unsqueeze(1) # 
        mix_embedding = mix_embedding.repeat(1, x.shape[1], 1)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        mix_embedding = self.EE_fusion(mix_embedding, embedding) # 使用神经网络进行融合
        # mix_embedding2 = selected_embeddings2.mean(1)
        #mix_embedding =  embedding + mix_embedding # 直接相加
        # new detection results
        # embedding_now = mix_embedding.unsqueeze(1)
        # embedding_now = embedding_now.repeat(1, x.shape[1], 1)
        f_now = self.detection.fusion(mix_embedding, x) 
        #f_now = torch.cat((x, embedding_now), dim=2) # 
        f_now, _ = self.detection.gru(f_now) #  x  torch.Size([16, 125, 256])
        f_now = self.detection.fc(f_now)
        decision_time_now = torch.softmax(self.detection.outputlayer(f_now), dim=2) # x  torch.Size([16, 125, 2])
        
        top_k = top_k.mean(1)  # get avg score,higher score will have more weight
        larger = top_k > self.tao
        top_k = top_k * larger
        top_k = top_k/2.0
        # print('top_k ',top_k)
        # assert 1==2
        # print('tok_k[ ',top_k.shape)
        # print('decision_time ',decision_time.shape)
        # print('decision_time_now ',decision_time_now.shape)
        neg_w = top_k.unsqueeze(1).unsqueeze(2)
        neg_w = neg_w.repeat(1, decision_time_now.shape[1], decision_time_now.shape[2])
        # print('neg_w ',neg_w.shape)
        #print('neg_w ',neg_w[:,0:10,0])
        pos_w = 1-neg_w
        #print('pos_w ',pos_w[:,0:10,0])
        decision_time_final = decision_time*pos_w + neg_w*decision_time_now
        #print('decision_time_final ',decision_time_final[0,0:10,0])
        # print(decision_time_final[0,:,:])
        #assert 1==2
        return decision_time_final
    
    def forward(self, x, ref, label=None):
        batch, time, dim = x.shape
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1)
        if self.att_pool == True:
            mean_embedding = self.bn(mean_embedding)
            embeddings = embeddings.transpose(1,2)
            embeddings = self.bn(embeddings)
            embeddings = embeddings.transpose(1,2)
            embedding = self.attention_pooling(embeddings, mean_embedding)
        else:
            embedding = mean_embedding
        if self.enhancement == True:
            decision_time = self.orcal_EE(x, embedding, label)
            decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
            return decision_time[:,:,0], decision_up, logit

        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding, x) 
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, logit
