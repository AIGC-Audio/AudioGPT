import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    # Audio feature extractor in AD-NeRF
    def __init__(self, in_dim=29, out_dim=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = out_dim
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(in_dim, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2,
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
        x = x[:, 8-half_w:8+half_w, :].permute(0, 2, 1) # [b,t=16,c]=>[b,c,t=16]
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


class NeRFBackbone(nn.Module):
    def __init__(self, pos_dim=3, cond_dim=64, view_dim=3, hid_dim=128, num_density_linears=8, num_color_linears=3, skip_layer_indices=[4]):
        super(NeRFBackbone, self).__init__()
        self.pos_dim = pos_dim
        self.view_dim = view_dim
        self.cond_dim = cond_dim
        self.hid_dim = hid_dim
        self.out_dim = 4 # rgb+sigma

        self.num_density_linears = num_density_linears
        self.num_color_linears = num_color_linears
        self.skip_layer_indices = skip_layer_indices # specify which layer in density_linears could get the raw input by skip connection

        density_input_dim = pos_dim + cond_dim
        self.density_linears = nn.ModuleList(
            [nn.Linear(density_input_dim, hid_dim)] + 
            [nn.Linear(hid_dim, hid_dim) if i not in self.skip_layer_indices else nn.Linear(hid_dim + density_input_dim, hid_dim) for i in range(num_density_linears-1)])
        self.density_out_linear = nn.Linear(hid_dim, 1)

        color_input_dim = view_dim + hid_dim
        self.color_linears = nn.ModuleList(
            [nn.Linear(color_input_dim, hid_dim//2)] + 
            [nn.Linear(hid_dim//2, hid_dim//2) for _ in range(num_color_linears-1)])
        self.color_out_linear = nn.Linear(hid_dim//2, 3)

    def forward(self, pos, cond, view):
        """
        pos: [bs, n_sample, pos_dim]; the encoding of xyz
        cond: [cond_dim,]; condition features
        view: [bs, view_dim]; the encoding of view direction
        """
        bs, n_sample, _ = pos.shape
        if cond.ndim == 1: # [cond_dim]
            cond = cond.squeeze()[None, None, :].expand([bs, n_sample, self.cond_dim])
        elif cond.ndim == 2: # [batch, cond_dim]
            cond = cond[:, None, :].expand([bs, n_sample, self.cond_dim])
        view = view[:, None, :].expand([bs, n_sample, self.view_dim])
        density_linear_input = torch.cat([pos, cond], dim=-1)
        h = density_linear_input
        for i in range(len(self.density_linears)):
            h = self.density_linears[i](h)
            h = F.relu(h)
            if i in self.skip_layer_indices:
                h = torch.cat([density_linear_input, h], -1)
        sigma = self.density_out_linear(h)  # [..., 1]

        h = torch.cat([h, view], -1)
        for i in range(len(self.color_linears)):
            h = self.color_linears[i](h)
            h = F.relu(h)
        rgb = self.color_out_linear(h)  # [..., 3]

        outputs = torch.cat([rgb, sigma], -1)  # [..., 4]
        return outputs


