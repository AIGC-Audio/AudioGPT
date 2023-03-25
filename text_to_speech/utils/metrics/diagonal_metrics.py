import torch


def get_focus_rate(attn, src_padding_mask=None, tgt_padding_mask=None):
    '''
    attn: bs x L_t x L_s
    '''
    if src_padding_mask is not None:
        attn = attn * (1 - src_padding_mask.float())[:, None, :]

    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    focus_rate = attn.max(-1).values.sum(-1)
    focus_rate = focus_rate / attn.sum(-1).sum(-1)
    return focus_rate


def get_phone_coverage_rate(attn, src_padding_mask=None, src_seg_mask=None, tgt_padding_mask=None):
    '''
    attn: bs x L_t x L_s
    '''
    src_mask = attn.new(attn.size(0), attn.size(-1)).bool().fill_(False)
    if src_padding_mask is not None:
        src_mask |= src_padding_mask
    if src_seg_mask is not None:
        src_mask |= src_seg_mask

    attn = attn * (1 - src_mask.float())[:, None, :]
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    phone_coverage_rate = attn.max(1).values.sum(-1)
    # phone_coverage_rate = phone_coverage_rate / attn.sum(-1).sum(-1)
    phone_coverage_rate = phone_coverage_rate / (1 - src_mask.float()).sum(-1)
    return phone_coverage_rate


def get_diagonal_focus_rate(attn, attn_ks, target_len, src_padding_mask=None, tgt_padding_mask=None,
                            band_mask_factor=5, band_width=50):
    '''
    attn: bx x L_t x L_s
    attn_ks: shape: tensor with shape [batch_size], input_lens/output_lens

    diagonal: y=k*x (k=attn_ks, x:output, y:input)
    1 0 0
    0 1 0
    0 0 1
    y>=k*(x-width) and y<=k*(x+width):1
    else:0
    '''
    # width = min(target_len/band_mask_factor, 50)
    width1 = target_len / band_mask_factor
    width2 = target_len.new(target_len.size()).fill_(band_width)
    width = torch.where(width1 < width2, width1, width2).float()
    base = torch.ones(attn.size()).to(attn.device)
    zero = torch.zeros(attn.size()).to(attn.device)
    x = torch.arange(0, attn.size(1)).to(attn.device)[None, :, None].float() * base
    y = torch.arange(0, attn.size(2)).to(attn.device)[None, None, :].float() * base
    cond = (y - attn_ks[:, None, None] * x)
    cond1 = cond + attn_ks[:, None, None] * width[:, None, None]
    cond2 = cond - attn_ks[:, None, None] * width[:, None, None]
    mask1 = torch.where(cond1 < 0, zero, base)
    mask2 = torch.where(cond2 > 0, zero, base)
    mask = mask1 * mask2

    if src_padding_mask is not None:
        attn = attn * (1 - src_padding_mask.float())[:, None, :]
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    diagonal_attn = attn * mask
    diagonal_focus_rate = diagonal_attn.sum(-1).sum(-1) / attn.sum(-1).sum(-1)
    return diagonal_focus_rate, mask
