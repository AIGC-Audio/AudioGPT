import torch
from torch import nn
import torch.nn.functional as F


class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        # Fix the highway input if necessary
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x):
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x)  # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]


class TacotronEncoder(nn.Module):
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims, embed_dims, embed_dims, dropout=dropout)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)
        self.proj_out = nn.Linear(cbhg_channels * 2, cbhg_channels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        x = self.proj_out(x)
        return x


class RNNEncoder(nn.Module):
    def __init__(self, num_chars, embedding_dim, n_convolutions=3, kernel_size=5):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=0)
        convolutions = []
        for _ in range(n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(embedding_dim,
                         embedding_dim,
                         kernel_size=kernel_size, stride=1,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(embedding_dim, int(embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        input_lengths = (x > 0).sum(-1)
        input_lengths = input_lengths.cpu().numpy()

        x = self.embedding(x)
        x = x.transpose(1, 2)  # [B, H, T]
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training) + x
        x = x.transpose(1, 2)  # [B, T, H]

        # pytorch tensor are not reversible, hence the conversion
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, decoder_rnn_dim, dropout):
        super(DecoderRNN, self).__init__()
        self.in_conv1d = nn.Sequential(
            torch.nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=9, padding=4,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=9, padding=4,
            ),
        )
        self.ln = nn.LayerNorm(hidden_size)
        if decoder_rnn_dim == 0:
            decoder_rnn_dim = hidden_size * 2
        self.rnn = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=decoder_rnn_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.rnn.flatten_parameters()
        self.conv1d = torch.nn.Conv1d(
            in_channels=decoder_rnn_dim * 2,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        input_masks = x.abs().sum(-1).ne(0).data[:, :, None]
        input_lengths = input_masks.sum([-1, -2])
        input_lengths = input_lengths.cpu().numpy()

        x = self.in_conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.ln(x)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # [B, T, C]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x * input_masks
        pre_mel = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        pre_mel = pre_mel * input_masks
        return pre_mel
