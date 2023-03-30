import os
import numpy as np
import torch
import tqdm
import cv2
import importlib
import math
from scipy.ndimage import gaussian_filter1d

from audio_to_face.inference.base_nerf_infer import BaseNeRFInfer
from audio_to_face.data_util.extract_mel import get_mel_from_fname
from audio_to_face.utils.commons.ckpt_utils import load_ckpt
from audio_to_face.utils.commons.hparams import hparams, set_hparams
from audio_to_face.utils.commons.tensor_utils import move_to_cuda, convert_to_tensor, convert_to_np
from audio_to_face.utils.commons.euler2rot import euler_trans_2_c2w, c2w_to_euler_trans
from audio_to_face.modules.postnet.lle import compute_LLE_projection, find_k_nearest_neighbors


class LM3dNeRFInfer(BaseNeRFInfer):

    def get_cond_from_input(self, inp):
        """
        :param inp: {'audio_source_name': (str), 'cond_name': (str, optional)}
        :return: a list that contains the condition feature of NeRF
        """
        self.save_wav16k(inp)

        # load the lm3d as the condition for lm3d head nerf
        assert inp['cond_name'].endswith('.npy')
        lm3d_arr = np.load(inp['cond_name'])[0] # [T, w=16, c=29]
        idexp_lm3d = torch.from_numpy(lm3d_arr).float()
        print(f"Loaded pre-extracted 3D landmark sequence from {inp['cond_name']}!")
        
        # load the deepspeech features as the condition for lm3d torso nerf
        wav16k_name = self.wav16k_name
        deepspeech_name = wav16k_name[:-4] + '_deepspeech.npy'
        if not os.path.exists(deepspeech_name):
            print(f"Try to extract deepspeech from {wav16k_name}...")
            # deepspeech_python = '/home/yezhenhui/anaconda3/envs/audio_to_face/bin/python' # the path of your python interpreter that has installed DeepSpeech
            # extract_deepspeech_cmd = f'{deepspeech_python} data_util/deepspeech_features/extract_ds_features.py --input={wav16k_name} --output={deepspeech_name}'
            extract_deepspeech_cmd = f'python data_util/deepspeech_features/extract_ds_features.py --input={wav16k_name} --output={deepspeech_name}'
            os.system(extract_deepspeech_cmd)
            print(f"Saved deepspeech features of {wav16k_name} to {deepspeech_name}.")
        else:
            print(f"Try to load pre-extracted deepspeech from {deepspeech_name}...")
        deepspeech_arr = np.load(deepspeech_name) # [T, w=16, c=29]
        print(f"Loaded deepspeech features from {deepspeech_name}.")
        # get window condition of deepspeech
        from audio_to_face.data_gen.nerf.binarizer import get_win_conds
        num_samples = min(len(lm3d_arr), len(deepspeech_arr), self.infer_max_length)
        samples = [{} for _ in range(num_samples)]
        for idx, sample in enumerate(samples):
            sample['deepspeech_win'] = torch.from_numpy(deepspeech_arr[idx]).float().unsqueeze(0) # [B=1, w=16, C=29]
            sample['deepspeech_wins'] = torch.from_numpy(get_win_conds(deepspeech_arr, idx, smo_win_size=8)).float() # [W=8, w=16, C=29]
        
        idexp_lm3d_mean = self.dataset.idexp_lm3d_mean
        idexp_lm3d_std = self.dataset.idexp_lm3d_std
        idexp_lm3d_normalized = (idexp_lm3d.reshape([-1,68,3]) - idexp_lm3d_mean)/idexp_lm3d_std

        # step1. clamp the lm3d, to regularize apparent outliers
        lm3d_clamp_std = hparams['infer_lm3d_clamp_std']
        idexp_lm3d_normalized[:,0:17] = torch.clamp(idexp_lm3d_normalized[:,0:17], -lm3d_clamp_std, lm3d_clamp_std) # yaw_x_y_z
        idexp_lm3d_normalized[:,17:27,0:2] = torch.clamp(idexp_lm3d_normalized[:,17:27,0:2], -lm3d_clamp_std/2, lm3d_clamp_std/2) # brow_x_y
        idexp_lm3d_normalized[:,17:27,2] = torch.clamp(idexp_lm3d_normalized[:,17:27,2], -lm3d_clamp_std, lm3d_clamp_std) # brow_z
        idexp_lm3d_normalized[:,27:36] = torch.clamp(idexp_lm3d_normalized[:,27:36], -lm3d_clamp_std, lm3d_clamp_std) # nose
        idexp_lm3d_normalized[:,36:48,0:2] = torch.clamp(idexp_lm3d_normalized[:,36:48,0:2], -lm3d_clamp_std/2, lm3d_clamp_std/2) # eye_x_y
        idexp_lm3d_normalized[:,36:48,2] = torch.clamp(idexp_lm3d_normalized[:,36:48,2], -lm3d_clamp_std, lm3d_clamp_std) # eye_z
        idexp_lm3d_normalized[:,48:68] = torch.clamp(idexp_lm3d_normalized[:,48:68], -lm3d_clamp_std, lm3d_clamp_std) # mouth
        idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1,68*3])

        # step2. LLE projection to drag the predicted lm3d closer to the GT lm3d
        LLE_percent = hparams['infer_lm3d_lle_percent']
        if LLE_percent > 0:
            idexp_lm3d_normalized_database = torch.stack([s['idexp_lm3d_normalized'] for s in self.dataset.samples]).reshape([-1, 68*3])
            feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d_normalized[:, :48*3], feat_database=idexp_lm3d_normalized_database[:, :48*3], K=10)
            idexp_lm3d_normalized[:, :48*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d_normalized[:,:48*3]
        
        # step3. inject eye blink
        inject_eye_blink_mode = hparams.get("infer_inject_eye_blink_mode", "none")
        print(f"The eye blink mode is: {inject_eye_blink_mode}")
        if inject_eye_blink_mode == 'none':
            pass
        elif inject_eye_blink_mode == 'period':
            # get a eye blink period (~40 frames) from the gt data
            # then repeat it to the whole sequence length
            blink_ref_frames_start_idx = hparams["infer_eye_blink_ref_frames_start_idx"] # the index of start frame of a blink period,
            blink_ref_frames_end_idx = hparams["infer_eye_blink_ref_frames_end_idx"] # the index of end frame of a blink period,
            assert blink_ref_frames_start_idx != '' or blink_ref_frames_end_idx != '', "If you want to use `period` eye blink mode, please find a eye blink period in your GT frames, then set `infer_eye_blink_pattern_start_idx` in your config file"
            idexp_lm3d_normalized_database = torch.stack([s['idexp_lm3d_normalized'] for s in self.dataset.samples]).reshape([-1, 68*3])
            blink_eye_pattern = idexp_lm3d_normalized_database[blink_ref_frames_start_idx:blink_ref_frames_end_idx+1, 17*3:48*3].clone()
            repeated_blink_eye_pattern = blink_eye_pattern.repeat([len(idexp_lm3d_normalized)//len(blink_eye_pattern)+1,1])[:len(idexp_lm3d_normalized)]
            idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1, 68*3])
            idexp_lm3d_normalized[:, 17*3:48*3] = repeated_blink_eye_pattern
            idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1, 68,3])
        elif inject_eye_blink_mode == 'gt':
            # use the eye blink sequence from the gt data
            idexp_lm3d_normalized_database = torch.stack([s['idexp_lm3d_normalized'] for s in self.dataset.samples]).reshape([-1, 68*3])
            blink_eye_pattern = idexp_lm3d_normalized_database[:, 17*3:48*3].clone()
            repeated_blink_eye_pattern = blink_eye_pattern.repeat([len(idexp_lm3d_normalized)//len(blink_eye_pattern)+1,1])[:len(idexp_lm3d_normalized)]
            idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1, 68*3])
            idexp_lm3d_normalized[:, 17*3:48*3] = repeated_blink_eye_pattern
            idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1, 68,3])

        else:
            raise NotImplementedError()
        
        # step4. close the mouth in silent frames
        # todo: remove `infer_sil_ref_frame_idx`, close the mouth using the current frame instead.
        if hparams.get('infer_close_mouth_when_sil', False):
            idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1, 68*3])
            mel, energy = get_mel_from_fname(self.wav16k_name, return_energy=True)
            energy = energy.reshape([-1])
            if len(energy) < 2*len(idexp_lm3d_normalized):
                energy = np.concatenate([energy] + [energy[-1:]]*(2*len(idexp_lm3d_normalized)-len(energy)))
            energy = energy[:2*len(idexp_lm3d_normalized)]
            energy = energy.reshape([-1,2]).max(axis=1) # downsample with max_pool
            is_sil_mask = energy < 1e-5
            sil_index = np.where(is_sil_mask)[0]
            sil_ref_frame_idx = hparams['infer_sil_ref_frame_idx']
            assert sil_ref_frame_idx != '', "Please set `infer_sil_ref_frame_idx` to the index of a frame with closed mouth in the GT dataset"
            idexp_lm3d_normalized_database = torch.stack([s['idexp_lm3d_normalized'] for s in self.dataset.samples]).reshape([-1, 68*3])
            sil_mouth_pattern = idexp_lm3d_normalized_database[sil_ref_frame_idx, 48*3:68*3].clone()
            repeated_sil_mouth_pattern = sil_mouth_pattern.unsqueeze(0).repeat([len(sil_index),1])
            idexp_lm3d_normalized[sil_index, 48*3:68*3] = repeated_sil_mouth_pattern

        # step5. gaussian filter to smooth the whole sequence
        lm3d_smooth_sigma = hparams['infer_lm3d_smooth_sigma']
        if lm3d_smooth_sigma > 0:
            idexp_lm3d_normalized[:, :48*3] = convert_to_tensor(gaussian_filter1d(idexp_lm3d_normalized[:, :48*3].numpy(), sigma=lm3d_smooth_sigma))
            # idexp_lm3d_normalized = convert_to_tensor(gaussian_filter1d(idexp_lm3d_normalized.numpy(), sigma=lm3d_smooth_sigma))
        
        idexp_lm3d_normalized_numpy = idexp_lm3d_normalized.cpu().numpy()
        idexp_lm3d_normalized_win_numpy = np.stack([get_win_conds(idexp_lm3d_normalized_numpy, i, smo_win_size=hparams['cond_win_size'], pad_option='edge') for i in range(idexp_lm3d_normalized_numpy.shape[0])])
        idexp_lm3d_normalized_win = torch.from_numpy(idexp_lm3d_normalized_win_numpy)

        for idx, sample in enumerate(samples):
            sample['cond'] = idexp_lm3d_normalized[idx].unsqueeze(0)
            if hparams['use_window_cond']:
                sample['cond_win'] = idexp_lm3d_normalized_win[idx]
                sample['cond_wins'] = torch.from_numpy(get_win_conds(idexp_lm3d_normalized_win_numpy, idx, hparams['smo_win_size'], 'edge'))
        return samples


if __name__ == '__main__':
    from audio_to_face.utils.commons.hparams import set_hparams
    from audio_to_face.utils.commons.hparams import hparams as hp
    inp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'cond_name': 'infer_out/May/pred_lm3d/zozo.npy',
            'out_video_name': 'infer_out/May/pred_video/zozo.mp4',
            }

    LM3dNeRFInfer.example_run(inp)