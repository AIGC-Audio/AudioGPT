import torch
from scipy.spatial.transform import Rotation as R
from audio_to_face.utils.commons.tensor_utils import convert_to_tensor


def rot2euler(rot, use_radian=True):
    r = R.from_matrix(rot)
    return r.as_euler('xyz', degrees=not use_radian)

def euler2rot(euler, use_radian=True):
    r = R.from_euler('xyz',euler, degrees=not use_radian)
    return r.as_matrix()

def c2w_to_euler_trans(c2w):
    if c2w.ndim == 3:
        e = rot2euler(c2w[:, :3, :3]) # [B, 3]
        t = c2w[:, :3, 3].reshape([-1, 3])
    else:
        e = rot2euler(c2w[:3, :3]) # [B, 3]
        t = c2w[:3, 3].reshape([3])
    return e, t # [3+3]

def euler_trans_2_c2w(euler, trans):
    if euler.ndim == 2:
        rot = euler2rot(euler) # [b, 3, 3]
        bs = trans.shape[0]
        trans = trans.reshape([bs, 3, 1])
        rot = convert_to_tensor(rot).float()
        trans = convert_to_tensor(trans).float()
        c2w = torch.cat([rot, trans], dim=-1) # [b, 3, 4]
    else:
        rot = euler2rot(euler) # [3, 3]
        trans = trans.reshape([3, 1])
        rot = convert_to_tensor(rot).float()
        trans = convert_to_tensor(trans).float()
        c2w = torch.cat([rot, trans], dim=-1) # [3, 4]
    return c2w