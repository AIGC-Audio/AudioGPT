import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .film import Film

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        super(ConvBlock, self).__init__()

        self.activation = activation
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        x = act(self.bn1(self.conv1(x)), self.activation)
        x = act(self.bn2(self.conv2(x)), self.activation)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample, activation, momentum):
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block(x)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample, activation, momentum):
        super(DecoderBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv_block2 = ConvBlock(
            out_channels * 2, out_channels, kernel_size, activation, momentum
        )

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn)

    def prune(self, x):
        """Prune the shape of x after transpose convolution."""
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        x = x[
            :,
            :,
            padding[0] : padding[0] - self.stride[0],
            padding[1] : padding[1] - self.stride[1]]
        return x

    def forward(self, input_tensor, concat_tensor):
        x = act(self.bn1(self.conv1(input_tensor)), self.activation)
        # from IPython import embed; embed(using=False); os._exit(0)
        # x = self.prune(x)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        return x


class EncoderBlockRes1B(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes1B, self).__init__()
        size = (3,3)

        self.conv_block1 = ConvBlockRes(in_channels, out_channels, size, activation, momentum)
        self.conv_block2 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class DecoderBlockRes1B(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes1B, self).__init__()
        size = (3,3)
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=size, stride=stride,
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(out_channels * 2, out_channels, size, activation, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block5 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution.
        """
        if(both): x = x[:, :, 0 : - 1, 0:-1]
        else: x = x[:, :, 0: - 1, :]
        return x

    def forward(self, input_tensor, concat_tensor,both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x,both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class EncoderBlockRes2BCond(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum, cond_embedding_dim):
        super(EncoderBlockRes2BCond, self).__init__()
        size = (3, 3)

        self.conv_block1 = ConvBlockResCond(in_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block2 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.downsample = downsample

    def forward(self, x, cond_vec):
        encoder = self.conv_block1(x, cond_vec)
        encoder = self.conv_block2(encoder, cond_vec)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class DecoderBlockRes2BCond(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum, cond_embedding_dim):
        super(DecoderBlockRes2BCond, self).__init__()
        size = (3, 3)
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=size, stride=stride,
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockResCond(out_channels * 2, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block3 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution.
        """
        if(both): x = x[:, :, 0 : - 1, 0:-1]
        else: x = x[:, :, 0: - 1, :]
        return x

    def forward(self, input_tensor, concat_tensor, cond_vec, both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x, both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, cond_vec)
        x = self.conv_block3(x, cond_vec)
        return x

class EncoderBlockRes4BCond(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum, cond_embedding_dim):
        super(EncoderBlockRes4B, self).__init__()
        size = (3,3)

        self.conv_block1 = ConvBlockResCond(in_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block2 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block3 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block4 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.downsample = downsample

    def forward(self, x, cond_vec):
        encoder = self.conv_block1(x, cond_vec)
        encoder = self.conv_block2(encoder, cond_vec)
        encoder = self.conv_block3(encoder, cond_vec)
        encoder = self.conv_block4(encoder, cond_vec)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class DecoderBlockRes4BCond(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum, cond_embedding_dim):
        super(DecoderBlockRes4B, self).__init__()
        size = (3, 3)
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=size, stride=stride,
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockResCond(out_channels * 2, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block3 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block4 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)
        self.conv_block5 = ConvBlockResCond(out_channels, out_channels, size, activation, momentum, cond_embedding_dim)

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution.
        """
        if(both): x = x[:, :, 0 : - 1, 0:-1]
        else: x = x[:, :, 0: - 1, :]
        return x

    def forward(self, input_tensor, concat_tensor, cond_vec, both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x,both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, cond_vec)
        x = self.conv_block3(x, cond_vec)
        x = self.conv_block4(x, cond_vec)
        x = self.conv_block5(x, cond_vec)
        return x

class EncoderBlockRes4B(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes4B, self).__init__()
        size = (3, 3)

        self.conv_block1 = ConvBlockRes(in_channels, out_channels, size, activation, momentum)
        self.conv_block2 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class DecoderBlockRes4B(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes4B, self).__init__()
        size = (3,3)
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=size, stride=stride,
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(out_channels * 2, out_channels, size, activation, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block5 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution.
        """
        if(both): x = x[:, :, 0 : - 1, 0:-1]
        else: x = x[:, :, 0: - 1, :]
        return x

    def forward(self, input_tensor, concat_tensor,both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x,both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x

class ConvBlockResCond(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum, cond_embedding_dim):
        r"""Residual block.
        """
        super(ConvBlockResCond, self).__init__()

        self.activation = activation
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              dilation=(1, 1), padding=padding, bias=False)
        self.film1 = Film(channels=out_channels, cond_embedding_dim=cond_embedding_dim)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              dilation=(1, 1), padding=padding, bias=False)
        self.film2 = Film(channels=out_channels, cond_embedding_dim=cond_embedding_dim)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.film_res = Film(channels=out_channels, cond_embedding_dim=cond_embedding_dim)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x, cond_vec):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.film1(x, cond_vec)
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))
        x = self.film2(x, cond_vec)
        if self.is_shortcut:
            residual = self.shortcut(origin)
            residual = self.film_res(residual, cond_vec)
            return residual + x
        else:
            return origin + x

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        r"""Residual block.
        """
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              dilation=(1, 1), padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              dilation=(1, 1), padding=padding, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x

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

def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


def act(x, activation):
    if activation == 'relu':
        return F.relu_(x)

    elif activation == 'leaky_relu':
        return F.leaky_relu_(x, negative_slope=0.2)

    elif activation == 'swish':
        return x * torch.sigmoid(x)

    else:
        raise Exception('Incorrect activation!')