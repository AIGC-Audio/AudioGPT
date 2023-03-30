import numpy as np
import torch
import cv2
import os
import imageio


def to8b(x): 
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

def mse2psnr(x): 
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def img2mse(x, y): 
    return torch.mean((x - y) ** 2)

def video2images(video_name, out_dir):
    cap = cv2.VideoCapture(video_name)
    frame_num = 0
    while(True):
        _, frame = cap.read()
        if frame is None:
            break
        out_frame_name = os.path.join(out_dir, str(frame_num) + '.jpg')
        cv2.imwrite(out_frame_name, frame)
        frame_num += + 1
    cap.release()

def load_image_as_uint8_tensor(fname):
    """
    img: (H, W, 3) floatTensor
    """
    img = torch.as_tensor(imageio.imread(fname))
    return img

if __name__ =='__main__':
    video2images("test_data/May_val/AD-NeRF.mp4", "test_data/May_val/AD-NeRF")
    video2images("test_data/May_val/audio_to_face.mp4", "test_data/May_val/audio_to_face")
    video2images("test_data/May_val/GT.mp4", "test_data/May_val/GT")