import numpy as np

yaw_idx_in_mediapipe_mesh = [356, 454, 361, 288, 397, 379, 378, 377, 152, 148, 149, 150, 172,58, 132, 234, 127]
brow_idx_in_mediapipe_mesh = [70,  63, 105,  66, 107, 336, 296, 334, 293, 300]
nose_idx_in_mediapipe_mesh = [6, 5, 1, 2, 129, 240, 2, 460, 358]
eye_idx_in_mediapipe_mesh = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
mouth_idx_in_mediapipe_mesh = [61,  40,  37,   0, 267, 270, 291, 321, 314,  17,  84,  91,  78, 81,  13, 311, 308, 402,  14, 178]
lm68_idx_in_mediapipe_mesh = yaw_idx_in_mediapipe_mesh + brow_idx_in_mediapipe_mesh + nose_idx_in_mediapipe_mesh + eye_idx_in_mediapipe_mesh + mouth_idx_in_mediapipe_mesh

def mediapipe_lm478_to_face_alignment_lm68(lm478, H, W, return_2d=True):
    """
    lm478: [B, 478, 3] or [478,3]
    """
    lm478[..., 0] *= W
    lm478[..., 1] *= H
    n_dim = 2 if return_2d else 3
    if lm478.ndim == 2:
        return lm478[lm68_idx_in_mediapipe_mesh, :n_dim].astype(np.int16)
    elif lm478.ndim == 3:
        return lm478[:, lm68_idx_in_mediapipe_mesh, :n_dim].astype(np.int16)
    else:
        raise ValueError("input lm478 ndim should in 2 or 3!")