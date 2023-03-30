import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)



class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12, bias=False):
        super(ResidualBlock, self).__init__()

        if norm_type == 'bn':
            norm_builder = lambda: nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm_builder = lambda: nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
        else:
            norm_builder = lambda: nn.Identity()

        self.blocks = [
            nn.Sequential(
                norm_builder(),
                nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
                          padding=(dilation * (kernel_size - 1)) // 2, bias=bias),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                nn.GELU(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation, bias=bias),
            )
            for _ in range(n)
        ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return x


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, is_BTC=True, bias=False):
        super(ConvBlocks, self).__init__()
        self.is_BTC = is_BTC
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps, bias=bias)
              for d in dilations],
        )
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm = LayerNorm(channels, dim=1, eps=ln_eps)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(channels, out_dims, kernel_size=3, padding=1, bias=bias)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        if self.is_BTC:
            x = x.transpose(1, 2) # [B, C, T]
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class SeqLevelConvolutionalModel(nn.Module):
    def __init__(self, out_dim=64, dropout=0.5, audio_feat_type='ppg', backbone_type='unet', norm_type='bn'):
        nn.Module.__init__(self)
        self.audio_feat_type = audio_feat_type
        if audio_feat_type == 'ppg':
            self.audio_encoder = nn.Sequential(*[
                nn.Conv1d(29, 48, 3, 1, 1, bias=False),
                nn.BatchNorm1d(48) if norm_type=='bn' else LayerNorm(48, dim=1),
                nn.GELU(),
                nn.Conv1d(48, 48, 3, 1, 1, bias=False)
            ])  
            self.energy_encoder = nn.Sequential(*[
                nn.Conv1d(1, 16, 3, 1, 1, bias=False),
                nn.BatchNorm1d(16) if norm_type=='bn' else LayerNorm(16, dim=1),
                nn.GELU(),
                nn.Conv1d(16, 16, 3, 1, 1, bias=False)
            ]) 
        elif audio_feat_type == 'mel':
            self.mel_encoder = nn.Sequential(*[
                nn.Conv1d(80, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64) if norm_type=='bn' else LayerNorm(64, dim=1),
                nn.GELU(),
                nn.Conv1d(64, 64, 3, 1, 1, bias=False)
            ])  
        else:
            raise NotImplementedError("now only ppg or mel are supported!")

        self.style_encoder = nn.Sequential(*[
            nn.Linear(135, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        ]) 

        if backbone_type == 'resnet':
            self.backbone = ResNetBackbone()
        elif backbone_type == 'unet':
            self.backbone = UNetBackbone()
        elif backbone_type == 'resblocks':
            self.backbone = ResBlocksBackbone()
        else:
            raise NotImplementedError("Now only resnet and unet are supported!")

        self.out_layer = nn.Sequential(
            nn.BatchNorm1d(512) if norm_type=='bn' else LayerNorm(512, dim=1),
            nn.Conv1d(512, 64, 3, 1, 1, bias=False),
            nn.PReLU(),
            nn.Conv1d(64, out_dim, 3, 1, 1, bias=False)
        )
        self.feat_dropout = nn.Dropout(p=dropout)   

    @property
    def device(self):
        return self.backbone.parameters().__next__().device

    def forward(self, batch, ret, log_dict=None):
        style, x_mask = batch['style'].to(self.device), batch['x_mask'].to(self.device)
        style_feat = self.style_encoder(style)  # [B,C=135] => [B,C=128]

        if self.audio_feat_type == 'ppg':
            audio, energy = batch['audio'].to(self.device), batch['energy'].to(self.device)
            audio_feat = self.audio_encoder(audio.transpose(1,2)).transpose(1,2) * x_mask.unsqueeze(2)  # [B,T,C=29] => [B,T,C=48] 
            energy_feat = self.energy_encoder(energy.transpose(1,2)).transpose(1,2) * x_mask.unsqueeze(2)  # [B,T,C=1] => [B,T,C=16]
            feat = torch.cat([audio_feat, energy_feat], dim=2) # [B,T,C=48+16]
        elif self.audio_feat_type == 'mel':
            mel = batch['mel'].to(self.device)
            feat = self.mel_encoder(mel.transpose(1,2)).transpose(1,2) * x_mask.unsqueeze(2) # [B,T,C=64]
        
        feat, x_mask = self.backbone(x=feat, sty=style_feat, x_mask=x_mask)
        
        out = self.out_layer(feat.transpose(1,2)).transpose(1,2) * x_mask.unsqueeze(2)  # [B,T//2,C=256] => [B,T//2,C=64]
        
        ret['pred'] = out
        ret['mask'] = x_mask
        return out


class ResBlocksBackbone(nn.Module):
    def __init__(self, in_dim=64, out_dim=512, p_dropout=0.5, norm_type='bn'):
        super(ResBlocksBackbone,self).__init__()
        self.resblocks_0 = ConvBlocks(channels=in_dim, out_dims=64, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_1 = ConvBlocks(channels=64, out_dims=128, dilations=[1]*4, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_2 = ConvBlocks(channels=128, out_dims=256, dilations=[1]*14, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_3 = ConvBlocks(channels=512, out_dims=512, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_4 = ConvBlocks(channels=512, out_dims=out_dim, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
 
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x, scale_factor=0.5, mode='linear'))
        self.upsampler = LambdaLayer(lambda x: F.interpolate(x, scale_factor=4, mode='linear'))

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, sty, x_mask=1.):
        """
        x: [B, T, C]
        sty: [B, C=256]
        x_mask: [B, T]
        ret: [B, T/2, C]
        """
        x = x.transpose(1, 2)  # [B, C, T]
        x_mask = x_mask[:, None, :] # [B, 1, T]

        x = self.resblocks_0(x) * x_mask # [B, C, T]

        x_mask = self.downsampler(x_mask) # [B, 1, T/2]
        x = self.downsampler(x) * x_mask # [B, C, T/2]
        x = self.resblocks_1(x) * x_mask # [B, C, T/2]
        x = self.resblocks_2(x) * x_mask # [B, C, T/2]

        x = self.dropout(x.transpose(1,2)).transpose(1,2)
        sty = sty[:, :, None].repeat([1,1,x_mask.shape[2]]) # [B,C=256,T/2]
        x = torch.cat([x, sty], dim=1) # [B, C=256+256, T/2]

        x = self.resblocks_3(x) * x_mask # [B, C, T/2]
        x = self.resblocks_4(x) * x_mask # [B, C, T/2]

        x = x.transpose(1,2)
        x_mask = x_mask.squeeze(1)
        return x, x_mask



class ResNetBackbone(nn.Module):
    def __init__(self, in_dim=64, out_dim=512, p_dropout=0.5, norm_type='bn'):
        super(ResNetBackbone,self).__init__()
        self.resblocks_0 = ConvBlocks(channels=in_dim, out_dims=64, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_1 = ConvBlocks(channels=64, out_dims=128, dilations=[1]*4, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_2 = ConvBlocks(channels=128, out_dims=256, dilations=[1]*14, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_3 = ConvBlocks(channels=512, out_dims=512, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_4 = ConvBlocks(channels=512, out_dims=out_dim, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
 
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x, scale_factor=0.5, mode='linear'))
        self.upsampler = LambdaLayer(lambda x: F.interpolate(x, scale_factor=4, mode='linear'))

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, sty, x_mask=1.):
        """
        x: [B, T, C]
        sty: [B, C=256]
        x_mask: [B, T]
        ret: [B, T/2, C]
        """
        x = x.transpose(1, 2)  # [B, C, T]
        x_mask = x_mask[:, None, :] # [B, 1, T]

        x = self.resblocks_0(x) * x_mask # [B, C, T]

        x_mask = self.downsampler(x_mask) # [B, 1, T/2]
        x = self.downsampler(x) * x_mask # [B, C, T/2]
        x = self.resblocks_1(x) * x_mask # [B, C, T/2]

        x_mask = self.downsampler(x_mask) # [B, 1, T/4]
        x = self.downsampler(x) * x_mask # [B, C, T/4]
        x = self.resblocks_2(x) * x_mask # [B, C, T/4]

        x_mask = self.downsampler(x_mask) # [B, 1, T/8]
        x = self.downsampler(x) * x_mask # [B, C, T/8]
        x = self.dropout(x.transpose(1,2)).transpose(1,2)
        sty = sty[:, :, None].repeat([1,1,x_mask.shape[2]]) # [B,C=256,T/8]
        x = torch.cat([x, sty], dim=1) # [B, C=256+256, T/8]
        x = self.resblocks_3(x) * x_mask # [B, C, T/8]

        x_mask = self.upsampler(x_mask) # [B, 1, T/2]
        x = self.upsampler(x) * x_mask # [B, C, T/2]
        x = self.resblocks_4(x) * x_mask # [B, C, T/2]
        
        x = x.transpose(1,2)
        x_mask = x_mask.squeeze(1)
        return x, x_mask


class UNetBackbone(nn.Module):
    def __init__(self, in_dim=64, out_dim=512, p_dropout=0.5, norm_type='bn'):
        super(UNetBackbone, self).__init__()
        self.resblocks_0 = ConvBlocks(channels=in_dim, out_dims=64, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_1 = ConvBlocks(channels=64, out_dims=128, dilations=[1]*4, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_2 = ConvBlocks(channels=128, out_dims=256, dilations=[1]*8, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_3 = ConvBlocks(channels=512, out_dims=512, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False)
        self.resblocks_4 = ConvBlocks(channels=768, out_dims=512, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False) # [768 = c3(512) + c2(256)]
        self.resblocks_5 = ConvBlocks(channels=640, out_dims=out_dim, dilations=[1]*3, kernel_size=3, norm_type=norm_type, is_BTC=False) # [640 = c4(512) + c1(128)]

        self.downsampler = nn.Upsample(scale_factor=0.5, mode='linear')
        self.upsampler = nn.Upsample(scale_factor=2, mode='linear')
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, sty, x_mask=1.):
        """
        x: [B, T, C]
        sty: [B, C=256]
        x_mask: [B, T]
        ret: [B, T/2, C]
        """
        x = x.transpose(1, 2)  # [B, C, T]
        x_mask = x_mask[:, None, :] # [B, 1, T]

        x0 = self.resblocks_0(x) * x_mask # [B, C, T]

        x_mask = self.downsampler(x_mask) # [B, 1, T/2]
        x = self.downsampler(x0) * x_mask # [B, C, T/2]
        x1 = self.resblocks_1(x) * x_mask # [B, C, T/2]

        x_mask = self.downsampler(x_mask) # [B, 1, T/4]
        x = self.downsampler(x1) * x_mask # [B, C, T/4]
        x2 = self.resblocks_2(x) * x_mask # [B, C, T/4]

        x_mask = self.downsampler(x_mask) # [B, 1, T/8]
        x = self.downsampler(x2) * x_mask # [B, C, T/8]
        x = self.dropout(x.transpose(1,2)).transpose(1,2)
        sty = sty[:, :, None].repeat([1,1,x_mask.shape[2]]) # [B,C=256,T/8]
        x = torch.cat([x, sty], dim=1) # [B, C=256+256, T/8]
        x3 = self.resblocks_3(x) * x_mask # [B, C, T/8]

        x_mask = self.upsampler(x_mask) # [B, 1, T/4]
        x = self.upsampler(x3) * x_mask # [B, C, T/4]
        x = torch.cat([x, self.dropout(x2.transpose(1,2)).transpose(1,2)], dim=1) # 
        x4 = self.resblocks_4(x) * x_mask # [B, C, T/4]

        x_mask = self.upsampler(x_mask) # [B, 1, T/2]
        x = self.upsampler(x4) * x_mask # [B, C, T/2]
        x = torch.cat([x, self.dropout(x1.transpose(1,2)).transpose(1,2)], dim=1)
        x5 = self.resblocks_5(x) * x_mask # [B, C, T/2]

        x = x5.transpose(1,2)
        x_mask = x_mask.squeeze(1)
        return x, x_mask


if __name__ == '__main__':
    pass
