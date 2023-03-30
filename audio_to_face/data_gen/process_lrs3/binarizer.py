import os
import numpy as np
from scipy.misc import face
import torch
from tqdm import trange
import pickle
from copy import deepcopy

from audio_to_face.data_util.face3d_helper import Face3DHelper
from audio_to_face.utils.commons.indexed_datasets import IndexedDataset, IndexedDatasetBuilder


def load_video_npy(fn):
    assert fn.endswith(".npy")
    ret_dict = np.load(fn,allow_pickle=True).item()
    video_dict = {
        'coeff': ret_dict['coeff'], # [T, h]
        'lm68': ret_dict['lm68'], # [T, 68, 2]  
        'lm5': ret_dict['lm5'], # [T, 5, 2]
    }
    return video_dict

def cal_lm3d_in_video_dict(video_dict, face3d_helper):
    coeff = torch.from_numpy(video_dict['coeff']).float()
    identity = coeff[:, 0:80]
    exp = coeff[:, 80:144]
    idexp_lm3d = face3d_helper.reconstruct_idexp_lm3d(identity, exp).cpu().numpy()
    video_dict['idexp_lm3d'] = idexp_lm3d

def load_audio_npy(fn):
    assert fn.endswith(".npy")
    ret_dict = np.load(fn,allow_pickle=True).item()
    audio_dict = {
        "mel": ret_dict['mel'], # [T, 80]
        "f0": ret_dict['f0'], # [T,1]
    }
    return audio_dict


if __name__ == '__main__':
    face3d_helper = Face3DHelper(use_gpu=False)
    
    import glob,tqdm
    prefixs = ['val', 'train']
    binarized_ds_path = "data/binary/lrs3"
    os.makedirs(binarized_ds_path, exist_ok=True)
    for prefix in prefixs:
        databuilder = IndexedDatasetBuilder(os.path.join(binarized_ds_path, prefix), gzip=False)
        raw_base_dir =  '/home/yezhenhui/datasets/raw/lrs3_raw'
        spk_ids = sorted([dir_name.split("/")[-1] for dir_name in glob.glob(raw_base_dir + "/*")])
        spk_id2spk_idx = {spk_id : i for i,spk_id in enumerate(spk_ids) }
        np.save(os.path.join(binarized_ds_path, "spk_id2spk_idx.npy"), spk_id2spk_idx, allow_pickle=True)
        mp4_names = glob.glob(raw_base_dir + "/*/*.mp4")
        cnt = 0
        for i, mp4_name in tqdm.tqdm(enumerate(mp4_names), total=len(mp4_names)):
            if prefix == 'train':
                if i % 100 == 0:
                    continue
            else:
                if i % 100 != 0:
                    continue
            lst = mp4_name.split("/")
            spk_id = lst[-2]
            clip_id = lst[-1][:-4]
            audio_npy_name = os.path.join(raw_base_dir, spk_id, clip_id+"_audio.npy")
            hubert_npy_name = os.path.join(raw_base_dir, spk_id, clip_id+"_hubert.npy")
            video_npy_name = os.path.join(raw_base_dir, spk_id, clip_id+"_coeff_pt.npy")
            if (not os.path.exists(audio_npy_name)) or (not os.path.exists(video_npy_name)):
                print(f"Skip item for not found.")
                continue
            if (not os.path.exists(hubert_npy_name)):
                print(f"Skip item for hubert_npy not found.")
                continue
            audio_dict = load_audio_npy(audio_npy_name)
            hubert = np.load(hubert_npy_name)
            video_dict = load_video_npy(video_npy_name)
            cal_lm3d_in_video_dict(video_dict, face3d_helper)
            mel = audio_dict['mel']
            if mel.shape[0] < 64: # the video is shorter than 0.6s
                print(f"Skip item for too short.")
                continue
            audio_dict.update(video_dict)
            audio_dict['spk_id'] = spk_id
            audio_dict['spk_idx'] = spk_id2spk_idx[spk_id]
            audio_dict['item_id'] = spk_id + "_" + clip_id
            
            audio_dict['hubert'] = hubert # [T_x, hid=1024]
            databuilder.add_item(audio_dict)
            cnt += 1
        databuilder.finalize()
        print(f"{prefix} set has {cnt} samples!")