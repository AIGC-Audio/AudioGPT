import torch
import torch.nn as nn

class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class CNNPostNet(nn.Module):
    def __init__(self, in_out_dim=64):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.block1 = nn.Sequential(*[
            Conv1d(in_out_dim, 128, 3, 1, 1, False), # [B, T=9, C=]
            Conv1d(128, 128, 3, 1, 1, True),
            Conv1d(128, 128, 3, 1, 1, True),
        ])
        self.block2 = nn.Sequential(*[
            Conv1d(128, 256, 3, 1, 1, False), # [B, T=9, C=]
            Conv1d(256, 256, 3, 1, 1, True),
            Conv1d(256, 256, 3, 1, 1, True),
        ])
        self.block3 = nn.Sequential(*[
            Conv1d(256, 128, 3, 1, 1, residual=False),
            nn.Conv1d(128, in_out_dim, 1, 1, 0),
        ])

    def forward(self, x):
        nopadding_mask = ~ x.abs().sum(-1).eq(0).data # [B, T]
        diff_x = self.block1(x.transpose(1, 2)).transpose(1,2) * nopadding_mask.unsqueeze(2)
        diff_x = self.block2(diff_x.transpose(1, 2)).transpose(1,2) * nopadding_mask.unsqueeze(2)
        diff_x = self.block3(diff_x.transpose(1, 2)).transpose(1,2) * nopadding_mask.unsqueeze(2)
        refine_x = x + diff_x
        return refine_x


class PitchContourCNNPostNet(nn.Module):
    def __init__(self, in_out_dim=64, pitch_dim=32):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.block1 = nn.Sequential(*[
            Conv1d(in_out_dim+pitch_dim, 128, 3, 1, 1, False), # [B, T=9, C=]
            Conv1d(128, 128, 3, 1, 1, True),
            Conv1d(128, 128, 3, 1, 1, True),
        ])
        self.block2 = nn.Sequential(*[
            Conv1d(128, 256, 3, 1, 1, False), # [B, T=9, C=]
            Conv1d(256, 256, 3, 1, 1, True),
            Conv1d(256, 256, 3, 1, 1, True),
        ])
        self.block3 = nn.Sequential(*[
            Conv1d(256, 128, 3, 1, 1, residual=False),
            nn.Conv1d(128, in_out_dim, 1, 1, 0),
        ])

    def forward(self, x, pitch):
        nopadding_mask = ~ x.abs().sum(-1).eq(0).data # [B, T]
        inp = torch.cat([x,pitch],dim=-1)
        diff_x = self.block1(inp.transpose(1, 2)).transpose(1,2) * nopadding_mask.unsqueeze(2)
        diff_x = self.block2(diff_x.transpose(1, 2)).transpose(1,2) * nopadding_mask.unsqueeze(2)
        diff_x = self.block3(diff_x.transpose(1, 2)).transpose(1,2) * nopadding_mask.unsqueeze(2)
        refine_x = x + diff_x
        return refine_x


class MLPDiscriminator(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.backbone = nn.Sequential(*[
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.25, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1, bias=False)
        ])
    def forward(self, x):
        x_mask = x.sum(-1).ne(0) # [b, T]
        x_flatten = x[x_mask.unsqueeze(2).repeat([1,1,self.in_dim])].reshape([-1,self.in_dim])
        validity = self.backbone(x_flatten)
        return [validity]

if __name__ == '__main__':
    net = CNNPostNet()
    x = torch.rand(2, 9, 64)
    y = net(x)
    print(y.shape)