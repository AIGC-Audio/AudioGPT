import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import tqdm
from audio_to_face.modules.audio2pose.gmm_utils import Sample_GMM
from audio_to_face.utils.commons.tensor_utils import convert_to_tensor

class Audio2PoseModel(nn.Module):
    def __init__(self, recept_field=100):
        super(Audio2PoseModel, self).__init__()
        self.audio_encoder = nn.Sequential(
            # nn.Linear(in_features=1024*2, out_features=256),
            nn.Linear(in_features=2*29, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256)
        )
        self.backbone = WaveNet()
        # self.recept_field = 30
        self.recept_field = recept_field
    
    def forward(self, audio, history_pose_velocity):
        """
        audio: a fixed window of audio representation, [b, t=30, c=512]
        history_pose_velocity: [b, t=30, c=12]
        pred_pose_velocity_params: the GMM params of pose_and_velocity at t+1 steps, [b, c=12*2+1]
        """
        audio = self.audio_encoder(audio)
        ret = self.backbone(history_pose_velocity, audio) # [b, t, c]
        # pred_pose_velocity_params = ret[:, -1, :] # [b, c]
        # return pred_pose_velocity_params
        return ret

    def autoregressive_infer(self, long_audio, init_pose=None):
        """
        long_audio: [T, c=512]
        init_pose: euler_trans, [6,], note that trans is subtracted by mean_trans!    
        """
        n_frames = len(long_audio)
        pred_pose_and_velocity_lst = []

        audio_insert = long_audio[0:1].repeat([self.recept_field-1,1])
        long_audio = torch.cat([audio_insert, long_audio], dim=0)
        history_pose_and_velocity = torch.zeros([self.recept_field, 12]).float().to(long_audio.device)
        if init_pose is not None:
            init_pose = convert_to_tensor(init_pose).float().to(long_audio.device).unsqueeze(0).repeat([self.recept_field, 1]) # [self.recept_field, 6]
            history_pose_and_velocity[:,:6] = init_pose

        with torch.no_grad():
            for i in tqdm.tqdm(range(n_frames), desc='generating headpose'): 
                audio_window = long_audio[i: i+self.recept_field].unsqueeze(0) # [b=1, t=30, c=512]
                history_info = history_pose_and_velocity.unsqueeze(0) # [b=1, t=30, c=12]
                pred_pose_and_velocity_gmm_params = self.forward(audio_window, history_info)[:,-1,:] # [b=1, c=12*2+1]
                pred_pose_and_velocity = Sample_GMM(pred_pose_and_velocity_gmm_params.unsqueeze(1),ncenter=1,ndim=12,sigma_scale=0.0).to(long_audio.device) # [b=1,t=1,c=12]
                pred_pose_and_velocity_lst.append(pred_pose_and_velocity.cpu().squeeze()) # [c=12]
                history_pose_and_velocity = torch.cat([history_pose_and_velocity[1:,:], pred_pose_and_velocity.squeeze(0)],dim=0) # [29,c=12] + [1, c=12] ==> [30, c=12]
        pred_pose_and_velocity = torch.stack(pred_pose_and_velocity_lst) # [T, c=12]
        pred_pose = pred_pose_and_velocity[:,:6]
        return pred_pose


class WaveNet(nn.Module):
    ''' 
    We use WaveNet as the backbone of Audio2Pose model.

    Args:
        batch_size: number of batch size
        residual_layers: number of layers in each residual blocks
        residual_blocks: number of residual blocks
        dilation_channels: number of channels for the dilated convolution
        residual_channels: number of channels for the residual connections
        skip_channels: number of channels for the skip connections
        end_channels: number of channels for the end convolution
        classes: Number of possible values each sample can have as output
        kernel_size: size of dilation convolution kernel
        output_length(int): Number of samples that are generated for each input
        use_bias: whether bias is used in each layer.
        cond(bool): whether condition information are applied. if cond == True:
            cond_channels: channel number of condition information
        `` loss(str): GMM loss is adopted. ``
    '''
    def __init__(self,
                #  residual_layers = 7,
                 residual_layers = 3,
                 residual_blocks = 2,
                 dilation_channels = 128,
                 residual_channels = 128,
                 skip_channels = 256,
                 kernel_size = 2,
                 use_bias = True,
                 cond = True,
                 input_channels = 12,
                 ncenter = 1,
                 ndim = 12,
                 output_channels = (2*12+1)*1,
                 cond_channels = 256,
                 activation = 'leakyrelu'):
        super(WaveNet, self).__init__()
        
        self.layers = residual_layers
        self.blocks = residual_blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.ncenter = ncenter
        self.ndim = ndim
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        # self.output_length = output_length
        self.bias = use_bias
        self.cond = cond
        self.cond_channels = cond_channels
        
        # build modules
        self.dilations = []
        self.dilation_queues = []
        residual_blocks = []
        self.receptive_field = 1
        
        # 1x1 convolution to create channels
        self.start_conv1 = nn.Conv1d(in_channels=self.input_channels,
                                     out_channels=self.residual_channels,
                                     kernel_size=1,
                                     bias=True)
        self.start_conv2 = nn.Conv1d(in_channels=self.residual_channels,
                                     out_channels=self.residual_channels,
                                     kernel_size=1,
                                     bias=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        self.drop_out2D = nn.Dropout2d(p=0.5)
        
        # build residual blocks
        for b in range(self.blocks):
            new_dilation = 1
            additional_scope = kernel_size - 1
            for i in range(self.layers):
                # create current residual block
                residual_blocks.append(residual_block(dilation = new_dilation,
                                                      dilation_channels = self.dilation_channels,
                                                      residual_channels = self.residual_channels,
                                                      skip_channels = self.skip_channels,
                                                      kernel_size = self.kernel_size,
                                                      use_bias = self.bias,
                                                      cond = self.cond,
                                                      cond_channels = self.cond_channels))
                new_dilation *= 2
                
                self.receptive_field += additional_scope
                additional_scope *= 2
        
        self.residual_blocks = nn.ModuleList(residual_blocks)
        # end convolutions
        
        self.end_conv_1 = nn.Conv1d(in_channels = self.skip_channels,
                                    out_channels = self.output_channels,
                                    kernel_size = 1,
                                    bias = True)
        self.end_conv_2 = nn.Conv1d(in_channels = self.output_channels,
                                    out_channels = self.output_channels,
                                    kernel_size = 1,
                                    bias = True)
        
    
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
    
    def forward(self, inp, cond=None):
        '''
        Args:
            inp: [b, T, ndim]
            cond: [b, T, nfeature]
        Returns:
            res: [b, T, ndim]
        '''
        inp = inp.transpose(1, 2)
        if cond is not None:
            cond = cond.transpose(1, 2)
        # dropout
        x = self.drop_out2D(inp)
        
        # preprocess
        x = self.activation(self.start_conv1(x))
        x = self.activation(self.start_conv2(x))
        skip = 0
        for i, dilation_block in enumerate(self.residual_blocks):
            x, current_skip = self.residual_blocks[i](x, cond)
            skip += current_skip
        
        # postprocess
        res = self.end_conv_1(self.activation(skip))
        res = self.end_conv_2(self.activation(res))
        
        # cut the output size
        # res = res[:, :, -self.output_length:]  # [b, ndim, T] 
        res = res.transpose(1, 2)  # [b, T, ndim]
        return res
    

class residual_block(nn.Module):
    '''
    This is the implementation of a residual block in wavenet model. Every
    residual block takes previous block's output as input. The forward pass of 
    each residual block can be illusatrated as below:
        
    ######################### Current Residual Block ##########################
    #     |-----------------------*residual*--------------------|             #
    #     |                                                     |             # 
    #     |        |-- dilated conv -- tanh --|                 |             #
    # -> -|-- pad--|                          * ---- |-- 1x1 -- + --> *input* #
    #              |-- dilated conv -- sigm --|      |                        #
    #                                               1x1                       # 
    #                                                |                        # 
    # ---------------------------------------------> + -------------> *skip*  #
    ###########################################################################
    As shown above, each residual block returns two value: 'input' and 'skip':
        'input' is indeed this block's output and also is the next block's input.
        'skip' is the skip data which will be added finally to compute the prediction.
    The input args own the same meaning in the WaveNet class.
    
    '''
    def __init__(self,
                 dilation,
                 dilation_channels = 32,
                 residual_channels = 32,
                 skip_channels = 256,
                 kernel_size = 2,
                 use_bias = False,
                 cond = True,
                 cond_channels = 128):
        super(residual_block, self).__init__()
        
        self.dilation = dilation
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.bias = use_bias
        self.cond = cond
        self.cond_channels = cond_channels
        # zero padding to the left of the sequence.
        self.padding = (int((self.kernel_size - 1) * self.dilation), 0)
        
        # dilated convolutions
        self.filter_conv= nn.Conv1d(in_channels = self.residual_channels,
                                    out_channels = self.dilation_channels,
                                    kernel_size = self.kernel_size,
                                    dilation = self.dilation,
                                    bias = self.bias)
                
        self.gate_conv = nn.Conv1d(in_channels = self.residual_channels,
                                   out_channels = self.dilation_channels,
                                   kernel_size = self.kernel_size,
                                   dilation = self.dilation,
                                   bias = self.bias)
                
        # 1x1 convolution for residual connections
        self.residual_conv = nn.Conv1d(in_channels = self.dilation_channels,
                                       out_channels = self.residual_channels,
                                       kernel_size = 1,
                                       bias = self.bias)
                
        # 1x1 convolution for skip connections
        self.skip_conv = nn.Conv1d(in_channels = self.dilation_channels,
                                   out_channels = self.skip_channels,
                                   kernel_size = 1,
                                   bias = self.bias)
        
        # condition conv, no dilation
        if self.cond == True:
            self.cond_filter_conv = nn.Conv1d(in_channels = self.cond_channels,
                                    out_channels = self.dilation_channels,
                                    kernel_size = 1,
                                    bias = True)
            self.cond_gate_conv = nn.Conv1d(in_channels = self.cond_channels,
                                   out_channels = self.dilation_channels,
                                   kernel_size = 1,
                                   bias = True)
        
    
    def forward(self, inp, cond=None):
        if self.cond is True and cond is None:
            raise RuntimeError("set using condition to true, but no cond tensor inputed")
            
        x_pad = F.pad(inp, self.padding)
        # filter
        filt = self.filter_conv(x_pad)
        # gate
        gate = self.gate_conv(x_pad)
        
        if self.cond == True and cond is not None:
            filter_cond = self.cond_filter_conv(cond)
            gate_cond = self.cond_gate_conv(cond)
            # add cond results
            filt = filt + filter_cond
            gate = gate + gate_cond
                       
        # element-wise multiple
        filt = torch.tanh(filt)
        gate = torch.sigmoid(gate)
        x = filt * gate
        
        # residual and skip
        residual = self.residual_conv(x) + inp
        skip = self.skip_conv(x)
        return residual, skip
    

if __name__ == '__main__':
    audio2pose_model = Audio2PoseModel()
    audio = torch.rand([128, 512])
    pred_pose = audio2pose_model.autoregressive_infer(audio)
    print(pred_pose.shape)