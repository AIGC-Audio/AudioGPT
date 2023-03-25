import numpy as np
import torch
import torch.nn as nn


class SingleWindowDisc(nn.Module):
    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.model = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),            
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]),
        ])
        ds_size = (time_length // 2 ** 3, (freq_length + 7) // 2 ** 3)
        self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)

    def forward(self, x):
        """
        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        for l in self.model:
            x = l(x)
            h.append(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)  # [B, 1]
        return validity, h


class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.discriminators = nn.ModuleList()

        for time_length in time_lengths:
            self.discriminators += [SingleWindowDisc(time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size)]

    def forward(self, x, x_len, start_frames_wins=None):
        '''
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.discriminators)
        h = []
        for i, start_frames in zip(range(len(self.discriminators)), start_frames_wins):
            x_clip, start_frames = self.clip(x, x_len, self.win_lengths[i], start_frames)  # (B, win_length, C)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            x_clip, h_ = self.discriminators[i](x_clip)
            h += h_
            validity.append(x_clip)
        if len(validity) != len(self.discriminators):
            return None, start_frames_wins, h
        validity = sum(validity)  # [B]
        return validity, start_frames_wins, h

    def clip(self, x, x_len, win_length, start_frames=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        T_start = 0
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame: start_frame + win_length]
        return x_batch, start_frames


class Discriminator(nn.Module):
    def __init__(self, time_lengths=[32, 64, 128], freq_length=80, kernel=(3, 3), c_in=1,
                 hidden_size=128):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.discriminator = MultiWindowDiscriminator(
            freq_length=freq_length,
            time_lengths=time_lengths,
            kernel=kernel,
            c_in=c_in, hidden_size=hidden_size
        )


    def forward(self, x, start_frames_wins=None):
        """

        :param x: [B, T, 80]
        :param return_y_only:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :] # [B,1,T,80]
        x_len = x.sum([1, -1]).ne(0).int().sum([-1])
        ret = {'y_c': None, 'y': None}
        ret['y'], start_frames_wins, ret['h'] = self.discriminator(
            x, x_len, start_frames_wins=start_frames_wins)

        ret['start_frames_wins'] = start_frames_wins
        return ret

