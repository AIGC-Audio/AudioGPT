"""This script contains the image preprocessing code for Deep3DFaceRecon_pytorch
"""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2
import os
from skimage import transform as trans
import torch
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

# bounding box for 68 landmark detection
def BBRegression(points, params):

    w1 = params['W1']
    b1 = params['B1']
    w2 = params['W2']
    b2 = params['B2']
    data = points.copy()
    data = data.reshape([5, 2])
    data_mean = np.mean(data, axis=0)
    x_mean = data_mean[0]
    y_mean = data_mean[1]
    data[:, 0] = data[:, 0] - x_mean
    data[:, 1] = data[:, 1] - y_mean

    rms = np.sqrt(np.sum(data ** 2)/5)
    data = data / rms
    data = data.reshape([1, 10])
    data = np.transpose(data)
    inputs = np.matmul(w1, data) + b1
    inputs = 2 / (1 + np.exp(-2 * inputs)) - 1
    inputs = np.matmul(w2, inputs) + b2
    inputs = np.transpose(inputs)
    x = inputs[:, 0] * rms + x_mean
    y = inputs[:, 1] * rms + y_mean
    w = 224/inputs[:, 2] * rms
    rects = [x, y, w, w]
    return np.array(rects).reshape([4])

# utils for landmark detection
def img_padding(img, box):
    success = True
    bbox = box.copy()
    res = np.zeros([2*img.shape[0], 2*img.shape[1], 3])
    res[img.shape[0] // 2: img.shape[0] + img.shape[0] //
        2, img.shape[1] // 2: img.shape[1] + img.shape[1]//2] = img

    bbox[0] = bbox[0] + img.shape[1] // 2
    bbox[1] = bbox[1] + img.shape[0] // 2
    if bbox[0] < 0 or bbox[1] < 0:
        success = False
    return res, bbox, success

# utils for landmark detection
def crop(img, bbox):
    padded_img, padded_bbox, flag = img_padding(img, bbox)
    if flag:
        crop_img = padded_img[padded_bbox[1]: padded_bbox[1] +
                            padded_bbox[3], padded_bbox[0]: padded_bbox[0] + padded_bbox[2]]
        crop_img = cv2.resize(crop_img.astype(np.uint8),
                            (224, 224), interpolation=cv2.INTER_CUBIC)
        scale = 224 / padded_bbox[3]
        return crop_img, scale
    else:
        return padded_img, 0

# utils for landmark detection
def scale_trans(img, lm, t, s):
    imgw = img.shape[1]
    imgh = img.shape[0]
    M_s = np.array([[1, 0, -t[0] + imgw//2 + 0.5], [0, 1, -imgh//2 + t[1]]],
                   dtype=np.float32)
    img = cv2.warpAffine(img, M_s, (imgw, imgh))
    w = int(imgw / s * 100)
    h = int(imgh / s * 100)
    img = cv2.resize(img, (w, h))
    lm = np.stack([lm[:, 0] - t[0] + imgw // 2, lm[:, 1] -
                   t[1] + imgh // 2], axis=1) / s * 100

    left = w//2 - 112
    up = h//2 - 112
    bbox = [left, up, 224, 224]
    cropped_img, scale2 = crop(img, bbox)
    assert(scale2!=0)
    t1 = np.array([bbox[0], bbox[1]])

    # back to raw img s * crop + s * t1 + t2
    t1 = np.array([w//2 - 112, h//2 - 112])
    scale = s / 100
    t2 = np.array([t[0] - imgw/2, t[1] - imgh / 2])
    inv = (scale/scale2, scale * t1 + t2.reshape([2]))
    return cropped_img, inv

# utils for landmark detection
def align_for_lm(img, five_points):
    five_points = np.array(five_points).reshape([1, 10])
    params = loadmat('util/BBRegressorParam_r.mat')
    bbox = BBRegression(five_points, params)
    assert(bbox[2] != 0)
    bbox = np.round(bbox).astype(np.int32)
    crop_img, scale = crop(img, bbox)
    return crop_img, scale, bbox


# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """
    w0, h0 = img.size


    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new

# utils for face recognition model
def estimate_norm(lm_68p, H):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]

def estimate_norm_torch(lm_68p, H):
    lm_68p_ = lm_68p.detach().cpu().numpy()
    M = []
    for i in range(lm_68p_.shape[0]):
        M.append(estimate_norm(lm_68p_[i], H))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
    return M
