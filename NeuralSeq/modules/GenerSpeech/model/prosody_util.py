from torch import nn
import copy
import torch
from utils.hparams import hparams
from modules.GenerSpeech.model.wavenet import WN
import math

from modules.fastspeech.tts_modules import LayerNorm
import torch.nn.functional as F
from utils.tts_utils import group_hidden_by_segs, sequence_mask

from scipy.cluster.vq import kmeans2
from torch.nn import functional as F


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5,
                 print_vq_prob=False):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.print_vq_prob = print_vq_prob
        self.register_buffer('data_initialized', torch.zeros(1))
        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        B, T, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # [B*T_mel, N_vq]
        indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, T, D]
        :return: [B, T, D]
        """
        B, T, _ = x.shape
        M, D = self.embedding.size()
        if self.training and self.data_initialized.item() == 0:
            print('| running kmeans in VQVAE')  # data driven initialization for the embeddings
            x_flat = x.detach().reshape(-1, D)
            rp = torch.randperm(x_flat.size(0))
            kd = kmeans2(x_flat[rp].data.cpu().numpy(), self.n_embeddings, minit='points')
            self.embedding.copy_(torch.from_numpy(kd[0]))
            x_flat, quantized, indices = self.encode(x)
            encodings = F.one_hot(indices, M).float()
            self.ema_weight.copy_(torch.matmul(encodings.t(), x_flat))
            self.ema_count.copy_(torch.sum(encodings, dim=0))

        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B, T)

        if self.training and self.data_initialized.item() != 0:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        self.data_initialized.fill_(1)

        e_latent_loss = F.mse_loss(x, quantized.detach(), reduction='none')
        nonpadding = (x.abs().sum(-1) > 0).float()
        e_latent_loss = (e_latent_loss.mean(-1) * nonpadding).sum() / nonpadding.sum()
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.print_vq_prob:
            print("| VQ code avg_probs: ", avg_probs)
        return quantized, loss, indices, perplexity

class CrossAttenLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttenLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, local_emotion, emotion_key_padding_mask=None, forcing=False):
        # src: (Tph, B, 256) local_emotion: (Temo, B, 256) emotion_key_padding_mask: (B, Temo)
        if forcing:
            maxlength = src.shape[0]
            k = local_emotion.shape[0] / src.shape[0]
            lengths1 = torch.ceil(torch.tensor([i for i in range(maxlength)]).to(src.device) * k) + 1
            lengths2 = torch.floor(torch.tensor([i for i in range(maxlength)]).to(src.device) * k) - 1
            mask1 = sequence_mask(lengths1, local_emotion.shape[0])
            mask2 = sequence_mask(lengths2, local_emotion.shape[0])
            mask = mask1.float() - mask2.float()
            attn_emo = mask.repeat(src.shape[1], 1, 1) # (B, Tph, Temo)
            src2 = torch.matmul(local_emotion.permute(1, 2, 0), attn_emo.float().transpose(1, 2)).permute(2, 0, 1)
        else:
            src2, attn_emo = self.multihead_attn(src, local_emotion, local_emotion, key_padding_mask=emotion_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_emo


class ProsodyAligner(nn.Module):
    def __init__(self, num_layers, guided_sigma=0.3, guided_layers=None, norm=None):
        super(ProsodyAligner, self).__init__()
        self.layers = nn.ModuleList([CrossAttenLayer(d_model=hparams['hidden_size'], nhead=2) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else num_layers

    def forward(self, src, local_emotion, src_key_padding_mask=None, emotion_key_padding_mask=None, forcing=False):
        output = src
        guided_loss = 0
        attn_emo_list = []
        for i, mod in enumerate(self.layers):
            # output: (Tph, B, 256), global_emotion: (1, B, 256), local_emotion: (Temo, B, 256) mask: None, src_key_padding_mask: (B, Tph),
            # emotion_key_padding_mask: (B, Temo)
            output, attn_emo = mod(output, local_emotion, emotion_key_padding_mask=emotion_key_padding_mask, forcing=forcing)
            attn_emo_list.append(attn_emo.unsqueeze(1))
            # attn_emo: (B, Tph, Temo) attn: (B, Tph, Tph)
            if i < self.guided_layers and src_key_padding_mask is not None:
                s_length = (~src_key_padding_mask).float().sum(-1) # B
                emo_length = (~emotion_key_padding_mask).float().sum(-1)
                attn_w_emo = _make_guided_attention_mask(src_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)

                g_loss_emo = attn_emo * attn_w_emo  # N, L, S
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, attn_emo_list

def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(
        -((grid_y.float() / rolen - grid_x.float() / rilen) ** 2) / (2 * (sigma ** 2))
    )

class LocalStyleAdaptor(nn.Module):
    def __init__(self, hidden_size, num_vq_codes=64, padding_idx=0):
        super(LocalStyleAdaptor, self).__init__()
        self.encoder = ConvBlocks(80, hidden_size, [1] * 5, 5, dropout=hparams['vae_dropout'])
        self.n_embed = num_vq_codes
        self.vqvae = VQEmbeddingEMA(self.n_embed, hidden_size, commitment_cost=hparams['lambda_commit'])
        self.wavenet = WN(hidden_channels=80, gin_channels=80, kernel_size=3, dilation_rate=1, n_layers=4)
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size

    def forward(self, ref_mels, mel2ph=None, no_vq=False):
        """

        :param ref_mels: [B, T, 80]
        :return: [B, 1, H]
        """
        padding_mask = ref_mels[:, :, 0].eq(self.padding_idx).data
        ref_mels = self.wavenet(ref_mels.transpose(1, 2), x_mask=(~padding_mask).unsqueeze(1).repeat([1, 80, 1])).transpose(1, 2)
        if mel2ph is not None:
            ref_ph, _ = group_hidden_by_segs(ref_mels, mel2ph, torch.max(mel2ph))
        else:
            ref_ph = ref_mels
        prosody = self.encoder(ref_ph)
        if no_vq:
            return prosody
        z, vq_loss, vq_tokens, ppl = self.vqvae(prosody)
        vq_loss = vq_loss.mean()
        return z, vq_loss, ppl




class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2, 1)).transpose(2, 1)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12):
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
                          padding=(dilation * (kernel_size - 1)) // 2),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                nn.GELU(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
            )
            for i in range(n)
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


class Pad(nn.ZeroPad2d):
    def __init__(self, kernel_size, dilation):
        pad_total = dilation * (kernel_size - 1)
        begin = pad_total // 2
        end = pad_total - begin

        super(Pad, self).__init__((begin, end, begin, end))


class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""

    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = (dilation * (kernel_size - 1))

        if causal:
            super(ZeroTemporalPad, self).__init__((total_pad, 0))
        else:
            begin = total_pad // 2
            end = total_pad - begin
            super(ZeroTemporalPad, self).__init__((begin, end))


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True):
        super(ConvBlocks, self).__init__()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps)
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
        self.post_net1 = nn.Conv1d(channels, out_dims, kernel_size=3, padding=1)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        x = x.transpose(1, 2)
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        return x.transpose(1, 2)


class TextConvEncoder(ConvBlocks):
    def __init__(self, embed_tokens, channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True):
        super().__init__(channels, out_dims, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights)
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(channels)

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [B x T x C]
        }
        """
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        return super().forward(x)


class ConditionalConvBlocks(ConvBlocks):
    def __init__(self, channels, g_channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, is_BTC=True):
        super().__init__(channels, out_dims, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights)
        self.g_prenet = nn.Conv1d(g_channels, channels, 3, padding=1)
        self.is_BTC = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, g, x_mask):
        if self.is_BTC:
            x = x.transpose(1, 2)
            g = g.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2)
        x = x + self.g_prenet(g)
        x = x * x_mask

        if not self.is_BTC:
            x = x.transpose(1, 2)
        x = super(ConditionalConvBlocks, self).forward(x)  # input needs to be BTC
        if not self.is_BTC:
            x = x.transpose(1, 2)
        return x
