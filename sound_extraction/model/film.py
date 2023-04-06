import torch
import torch.nn as nn

class Film(nn.Module):
    def __init__(self, channels, cond_embedding_dim):
        super(Film, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(cond_embedding_dim, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data, cond_vec):
        """
        :param data: [batchsize, channels, samples] or [batchsize, channels, T, F] or [batchsize, channels, F, T]
        :param cond_vec: [batchsize, cond_embedding_dim]
        :return:
        """
        bias = self.linear(cond_vec)  # [batchsize, channels]
        if len(list(data.size())) == 3:
            data = data + bias[..., None]
        elif len(list(data.size())) == 4:
            data = data + bias[..., None, None]
        else:
            print("Warning: The size of input tensor,", data.size(), "is not correct. Film is not working.")
        return data