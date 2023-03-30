import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    # Audio feature extractor in AD-NeRF
    def __init__(self, in_dim=29, out_dim=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        if win_size == 1:
            strides = [1,1,1,1]
        elif win_size == 2:
            strides = [2,1,1,1]
        elif win_size in [3, 4]:
            strides = [2,2,1,1]
        elif win_size in [5, 8]:
            strides = [2,2,2,1]
        elif win_size == 16:
            strides = [2,2,2,2]
        else:
            raise ValueError("unsupported win_size")
        self.dim_aud = out_dim
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(in_dim, 32, kernel_size=3, stride=strides[0],
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=strides[1],
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=strides[2],
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=strides[3],
                      padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        """
        x: [batch, win=16, hid=29]
        return:
            [batch, out_dim=76]
        """
        half_w = int(self.win_size/2)
        x = x.permute(0, 2, 1) # [b,t=16,c]=>[b,c,t=16]
        x = self.encoder_conv(x).squeeze(-1) # [b, c=64, 1] => [b, c]
        x = self.encoder_fc1(x).squeeze() # [b,out_dim=76]
        return x

class AudioAttNet(nn.Module):
    # Audio feature attention-based smoother in AD-NeRF
    def __init__(self, in_out_dim=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.in_out_dim = in_out_dim
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.in_out_dim, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        x: [b=8, c]
        return:
            [c]
        """
        y = x[:, :self.in_out_dim].permute(1, 0).unsqueeze(0)  # [b, c] => [1, c, b]
        y = self.attentionConvNet(y) # [1,1,b]
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1) # [8, 1]
        smoothed_y = torch.sum(y*x, dim=0) # [8,1]*[8,c]=>[8,c]=>[c,]
        return smoothed_y

