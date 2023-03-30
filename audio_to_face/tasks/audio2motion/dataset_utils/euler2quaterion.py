import numpy as np
import torch
import math
import numba
from scipy.spatial.transform import Rotation as R

def euler2quaterion(euler, use_radian=True):
    """
    euler: np.array, [batch, 3]
    return: the quaterion, np.array, [batch, 4]
    """
    r = R.from_euler('xyz',euler, degrees=not use_radian)
    return r.as_quat()

def quaterion2euler(quat, use_radian=True):
    """
    quat: np.array, [batch, 4]
    return: the euler, np.array, [batch, 3]
    """
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=not use_radian)

def rot2quaterion(rot):
    r = R.from_matrix(rot)
    return r.as_quat()

def quaterion2rot(quat):
    r = R.from_quat(quat)
    return r.as_matrix()

if __name__ == '__main__':
    euler = np.array([89.999,89.999,89.999] * 100).reshape([100,3])
    q = euler2quaterion(euler, use_radian=False)
    e = quaterion2euler(q, use_radian=False)
    print(" ")
