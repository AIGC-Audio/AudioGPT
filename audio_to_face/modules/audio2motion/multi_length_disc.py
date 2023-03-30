import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_to_face.modules.audio2motion.cnn_models import LambdaLayer


class Discriminator1DFactory(nn.Module):
    def __init__(self, time_length, kernel_size=3, in_dim=1, hidden_size=128, norm_type='bn'):
        super(Discriminator1DFactory, self).__init__()
        padding = kernel_size // 2

        def discriminator_block(in_filters, out_filters, first=False):
            """
            Input: (B, c, T)
            Output:(B, c, T//2)
            """
            conv = nn.Conv1d(in_filters, out_filters, kernel_size, 2, padding)
            block = [
                conv,  # padding = kernel//2
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if norm_type == 'bn' and not first:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            if norm_type == 'in' and not first:
                block.append(nn.InstanceNorm1d(out_filters, affine=True))
            block = nn.Sequential(*block)
            return block

        if time_length >= 8:
            self.model = nn.ModuleList([
                discriminator_block(in_dim, hidden_size, first=True),
                discriminator_block(hidden_size, hidden_size),
                discriminator_block(hidden_size, hidden_size),
            ])
            ds_size = time_length // (2 ** 3)
        elif time_length == 3:
            self.model = nn.ModuleList([
                nn.Sequential(*[
                    nn.Conv1d(in_dim, hidden_size, 3, 1, 0),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.Conv1d(hidden_size, hidden_size, 1, 1, 0),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm1d(hidden_size, 0.8),
                    nn.Conv1d(hidden_size, hidden_size, 1, 1, 0),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.BatchNorm1d(hidden_size, 0.8)
                ])
            ])
            ds_size = 1
        elif time_length == 1:
            self.model = nn.ModuleList([
                nn.Sequential(*[
                    nn.Linear(in_dim, hidden_size),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                ])
            ])
            ds_size = 1

        self.adv_layer = nn.Linear(hidden_size * ds_size, 1)

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        for l in self.model:
            x = l(x)
            h.append(x)
        if x.ndim == 2:
            b, ct = x.shape
            use_sigmoid = True
        else:
            b, c, t = x.shape
            ct = c * t
            use_sigmoid = False
        x = x.view(b, ct)
        validity = self.adv_layer(x)  # [B, 1]
        if use_sigmoid:
            validity = torch.sigmoid(validity)
        return validity, h


class CosineDiscriminator1DFactory(nn.Module):
    def __init__(self, time_length, kernel_size=3, in_dim=1, hidden_size=128, norm_type='bn'):
        super().__init__()
        padding = kernel_size // 2

        def discriminator_block(in_filters, out_filters, first=False):
            """
            Input: (B, c, T)
            Output:(B, c, T//2)
            """
            conv = nn.Conv1d(in_filters, out_filters, kernel_size, 2, padding)
            block = [
                conv,  # padding = kernel//2
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if norm_type == 'bn' and not first:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            if norm_type == 'in' and not first:
                block.append(nn.InstanceNorm1d(out_filters, affine=True))
            block = nn.Sequential(*block)
            return block

        self.model1 = nn.ModuleList([
            discriminator_block(in_dim, hidden_size, first=True),
            discriminator_block(hidden_size, hidden_size),
            discriminator_block(hidden_size, hidden_size),
        ])

        self.model2 = nn.ModuleList([
            discriminator_block(in_dim, hidden_size, first=True),
            discriminator_block(hidden_size, hidden_size),
            discriminator_block(hidden_size, hidden_size),
        ])

        self.relu = nn.ReLU()
    def forward(self, x1, x2):
        """

        :param x1: [B, C, T]
        :param x2: [B, C, T]
        :return: validity: [B, 1], h: List of hiddens
        """
        h1, h2 = [], []
        for l in self.model1:
            x1 = l(x1)
            h1.append(x1)
        for l in self.model2:
            x2 = l(x2)
            h2.append(x1)
        b,c,t = x1.shape
        x1 = x1.view(b, c*t)
        x2 = x2.view(b, c*t)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        # x1 = F.normalize(x1, p=2, dim=1)
        # x2 = F.normalize(x2, p=2, dim=1)
        validity = F.cosine_similarity(x1, x2)    
        return validity, [h1,h2]


class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, cond_dim=80, in_dim=64, kernel_size=3, hidden_size=128, disc_type='standard', norm_type='bn', reduction='sum'):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.reduction = reduction
        self.disc_type = disc_type

        if cond_dim > 0:
            self.use_cond = True
            self.cond_proj_layers = nn.ModuleList()
            self.in_proj_layers = nn.ModuleList()
        else:
            self.use_cond = False

        self.conv_layers = nn.ModuleList()
        for time_length in time_lengths:
            conv_layer = [
                Discriminator1DFactory(
                    time_length, kernel_size, in_dim=64, hidden_size=hidden_size,
                    norm_type=norm_type) if self.disc_type == 'standard' 
                else CosineDiscriminator1DFactory(time_length, kernel_size, in_dim=64, 
                    hidden_size=hidden_size,norm_type=norm_type)
            ]
            self.conv_layers += conv_layer
            if self.use_cond:
                self.cond_proj_layers.append(nn.Linear(cond_dim, 64))
                self.in_proj_layers.append(nn.Linear(in_dim, 64))
    
    def clip(self, x, cond, x_len, win_length, start_frames=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B,  T, C).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        clip_from_same_frame = start_frames is None
        T_start = 0
        # T_end = x_len.max() - win_length
        T_end = x_len.min() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]


        if clip_from_same_frame:
            x_batch = x[:, start_frame: start_frame + win_length, :]
            c_batch = cond[:, start_frame: start_frame + win_length, :] if cond is not None else None
        else:
            x_lst = []
            c_lst = []
            for i, start_frame in enumerate(start_frames):
                x_lst.append(x[i, start_frame: start_frame + win_length, :])
                if cond is not None:
                    c_lst.append(cond[i, start_frame: start_frame + win_length, :])
            x_batch = torch.stack(x_lst, dim=0)
            if cond is None:
                c_batch = None
            else:
                c_batch = torch.stack(c_lst, dim=0)
        return x_batch, c_batch, start_frames

    def forward(self, x, x_len, cond=None, start_frames_wins=None):
        '''
        Args:
            x (tensor): input mel, (B, T, C).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.conv_layers)
        h = []
        for i, start_frames in zip(range(len(self.conv_layers)), start_frames_wins):
            x_clip, c_clip, start_frames = self.clip(
                x, cond, x_len, self.win_lengths[i], start_frames)  # (B, win_length, C)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            if self.disc_type == 'standard':
                if self.use_cond:
                    x_clip = self.in_proj_layers[i](x_clip)  # (B, T, C)
                    c_clip = self.cond_proj_layers[i](c_clip)
                    x_clip = x_clip + c_clip
                validity_pred, h_ = self.conv_layers[i](x_clip.transpose(1,2))
            elif self.disc_type == 'cosine':
                assert self.use_cond is True
                x_clip = self.in_proj_layers[i](x_clip)  # (B, T, C)
                c_clip = self.cond_proj_layers[i](c_clip)
                validity_pred, h_ = self.conv_layers[i](x_clip.transpose(1,2), c_clip.transpose(1,2))
            else:
                raise NotImplementedError

            h += h_
            validity.append(validity_pred)
        if len(validity) != len(self.conv_layers):
            return None, start_frames_wins, h
        if self.reduction == 'sum':
            validity = sum(validity)  # [B]
        elif self.reduction == 'stack':
            validity = torch.stack(validity, -1)  # [B, W_L]
        return validity, start_frames_wins, h


class Discriminator(nn.Module):
    def __init__(self, x_dim=80, y_dim=64, disc_type='standard', 
                uncond_disc=False, kernel_size=3, hidden_size=128, norm_type='bn', reduction='sum', time_lengths=(8,16,32)):
        """_summary_

        Args:
            time_lengths (list, optional): the list of  window size. Defaults to [32, 64, 128].
            x_dim (int, optional): the dim of audio features. Defaults to 80, corresponding to mel-spec.
            y_dim (int, optional): the dim of facial coeff. Defaults to 64, correspond to exp; other options can be 7(pose) or 71(exp+pose).
            kernel (tuple, optional): _description_. Defaults to (3, 3).
            c_in (int, optional): _description_. Defaults to 1.
            hidden_size (int, optional): _description_. Defaults to 128.
            norm_type (str, optional): _description_. Defaults to 'bn'.
            reduction (str, optional): _description_. Defaults to 'sum'.
            uncond_disc (bool, optional): _description_. Defaults to False.
        """
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.x_dim, self.y_dim = x_dim, y_dim
        self.disc_type = disc_type
        self.reduction = reduction
        self.uncond_disc = uncond_disc

        if uncond_disc:
            self.x_dim = 0
            cond_dim = 0

        else:
            cond_dim = 64
            self.mel_encoder = nn.Sequential(*[
                    nn.Conv1d(self.x_dim, 64, 3, 1, 1, bias=False),
                    nn.BatchNorm1d(64),
                    nn.GELU(),
                    nn.Conv1d(64, cond_dim, 3, 1, 1, bias=False)
                ]) 

        self.disc = MultiWindowDiscriminator(
            time_lengths=self.time_lengths,
            in_dim=self.y_dim,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            hidden_size=hidden_size, norm_type=norm_type,
            reduction=reduction,
            disc_type=disc_type
        )
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x.transpose(1,2), scale_factor=0.5, mode='nearest').transpose(1,2))
    
    @property
    def device(self):
        return self.disc.parameters().__next__().device

    def forward(self,x, batch, start_frames_wins=None):
        """

        :param x: [B, T, C]
        :param cond: [B, T, cond_size]
        :return:
        """
        x = x.to(self.device)
        if not self.uncond_disc:
            mel = self.downsampler(batch['mel'].to(self.device))
            mel_feat = self.mel_encoder(mel.transpose(1,2)).transpose(1,2)
        else:
            mel_feat = None
        x_len = x.sum(-1).ne(0).int().sum([1])
        disc_confidence, start_frames_wins, h = self.disc(x, x_len, mel_feat, start_frames_wins=start_frames_wins)
        return disc_confidence
    
