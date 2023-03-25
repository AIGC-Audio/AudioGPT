import torch
import torch.nn as nn


class BasicDiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicDiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),

        )

    def forward(self, x):
        return self.block(x)


class ResDiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResDiscriminatorBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

        self.shortcut1 = nn.utils.weight_norm(nn.Conv1d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=2,
        ))

        self.block2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

        self.shortcut2 = nn.utils.weight_norm(nn.Conv1d(
            out_channel,
            out_channel,
            kernel_size=1,
            stride=1,
        ))

    def forward(self, x):
        x1 = self.block1(x)
        x1 = x1 + self.shortcut1(x)
        return self.block2(x1) + self.shortcut2(x1)


class ResNet18Discriminator(nn.Module):
    def __init__(self, stft_channel, in_channel=64):
        super(ResNet18Discriminator, self).__init__()
        self.input = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(stft_channel, in_channel, kernel_size=7, stride=2, padding=1, )),
            nn.LeakyReLU(0.2, True),
        )
        self.df1 = BasicDiscriminatorBlock(in_channel, in_channel)
        self.df2 = ResDiscriminatorBlock(in_channel, in_channel * 2)
        self.df3 = ResDiscriminatorBlock(in_channel * 2, in_channel * 4)
        self.df4 = ResDiscriminatorBlock(in_channel * 4, in_channel * 8)

    def forward(self, x):
        x = self.input(x)
        x = self.df1(x)
        x = self.df2(x)
        x = self.df3(x)
        return self.df4(x)


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channel=64, fft_size=1024, hop_length=256, win_length=1024, window="hann_window"):
        super(FrequencyDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.stft_channel = fft_size // 2 + 1
        self.resnet_disc = ResNet18Discriminator(self.stft_channel, in_channel)

    def forward(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_length, self.win_length, self.window)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        x_real = self.resnet_disc(real)
        x_imag = self.resnet_disc(imag)

        return x_real, x_imag
