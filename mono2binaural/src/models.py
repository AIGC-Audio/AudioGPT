import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src.warping import GeometricTimeWarper, MonotoneTimeWarper
from src.utils import Net


class GeometricWarper(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.warper = GeometricTimeWarper(sampling_rate=sampling_rate)

    def _transmitter_mouth(self, view):
        # offset between tracking markers and real mouth position in the dataset
        mouth_offset = np.array([0.09, 0, -0.20])
        quat = view[:, 3:, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()
        # make sure zero-padded values are set to non-zero values (else scipy raises an exception)
        norms = scipy.linalg.norm(quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        quat = quat + eps_val[:, None]
        transmitter_rot_mat = R.from_quat(quat)
        transmitter_mouth = transmitter_rot_mat.apply(mouth_offset, inverse=True)
        transmitter_mouth = th.Tensor(transmitter_mouth).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()
        if view.is_cuda:
            transmitter_mouth = transmitter_mouth.cuda()
        return transmitter_mouth

    def _3d_displacements(self, view):
        transmitter_mouth = self._transmitter_mouth(view)
        # offset between tracking markers and ears in the dataset
        left_ear_offset = th.Tensor([0, -0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, -0.08, -0.22])
        right_ear_offset = th.Tensor([0, 0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, 0.08, -0.22])
        # compute displacements between transmitter mouth and receiver left/right ear
        displacement_left = view[:, 0:3, :] + transmitter_mouth - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] + transmitter_mouth - right_ear_offset[None, :, None]
        displacement = th.stack([displacement_left, displacement_right], dim=1)
        return displacement

    def _warpfield(self, view, seq_length):
        return self.warper.displacements2warpfield(self._3d_displacements(view), seq_length)

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        return self.warper(th.cat([mono, mono], dim=1), self._3d_displacements(view))


class Warpnet(nn.Module):
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__()
        self.layers = [nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2) for l in range(layers)]
        self.layers = nn.ModuleList(self.layers)
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        self.neural_warper = MonotoneTimeWarper()
        self.geometric_warper = GeometricWarper()

    def neural_warpfield(self, view, seq_length):
        warpfield = view
        for layer in self.layers:
            warpfield = F.pad(warpfield, pad=[1, 0])
            warpfield = F.relu(layer(warpfield))
        warpfield = self.linear(warpfield)
        warpfield = F.interpolate(warpfield, size=seq_length)
        return warpfield

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        geometric_warpfield = self.geometric_warper._warpfield(view, mono.shape[-1])
        neural_warpfield = self.neural_warpfield(view, mono.shape[-1])
        warpfield = geometric_warpfield + neural_warpfield
        # ensure causality
        warpfield = -F.relu(-warpfield) # the predicted warp
        warped = self.neural_warper(th.cat([mono, mono], dim=1), warpfield)
        return warped

class BinauralNetwork(Net):
    def __init__(self,
                 view_dim=7,
                 warpnet_layers=4,
                 warpnet_channels=64,
                 model_name='binaural_network',
                 use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.warper = Warpnet(warpnet_layers, warpnet_channels)
        if self.use_cuda:
            self.cuda()

    def forward(self, mono, view):
        '''
        :param mono: the input signal as a B x 1 x T tensor
        :param view: the receiver/transmitter position as a B x 7 x T tensor
        :return: out: the binaural output produced by the network
                 intermediate: a two-channel audio signal obtained from the output of each intermediate layer
                               as a list of B x 2 x T tensors
        '''
        # print('mono ', mono.shape)
        # print('view ', view.shape)
        warped = self.warper(mono, view)
        # print('warped ', warped.shape)
        return warped
