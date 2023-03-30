import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm, trange
import torch
import face_alignment
import deep_3drecon
from moviepy.editor import VideoFileClip
import copy
from audio_to_face.utils.commons.multiprocess_utils import multiprocess_run_tqdm, multiprocess_run
from audio_to_face.utils.commons.meters import Timer
from decord import VideoReader
from decord import cpu, gpu
from audio_to_face.utils.commons.face_alignment_utils import mediapipe_lm478_to_face_alignment_lm68
import mediapipe

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')
mp_face_mesh = mediapipe.solutions.face_mesh
face_reconstructor = deep_3drecon.Reconstructor()


def chunk(iterable, chunk_size):
    final_ret = []
    cnt = 0
    ret = []
    for record in iterable:
        if cnt == 0:
            ret = []
        ret.append(record)
        cnt += 1
        if len(ret) == chunk_size:
            final_ret.append(ret)
            ret = []
    if len(final_ret[-1]) != chunk_size:
        final_ret.append(ret)
    return final_ret

# landmark detection in Deep3DRecon
def lm68_2_lm5(in_lm):
    assert in_lm.ndim == 2
    # in_lm: shape=[68,2]
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    # 将上述特殊角点的数据取出，得到5个新的角点数据，拼接起来。
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    # 将第一个角点放在了第三个位置
    lm = lm[[1,2,0,3,4],:2]
    return lm

def extract_frames_job(fname):
    out_name=fname.replace(".mp4", "_coeff_pt.npy").replace("datasets/raw/cropped_clips", "datasets/processed/coeff")
    if os.path.exists(out_name):
        return None
    video_reader = VideoReader(fname, ctx=cpu(0))
    frame_rgb_lst = video_reader.get_batch(list(range(0,len(video_reader)))).asnumpy()
    return frame_rgb_lst

def extract_lms_mediapipe_job(frames):
    if frames is None:
        return None
    with mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5) as face_mesh:
        ldms_normed = []
        frame_i = 0
        frame_ids = []
        for i in range(len(frames)):
            # Convert the BGR image to RGB before processing.
            ret = face_mesh.process(frames[i])
            # Print and draw face mesh landmarks on the image.
            if not ret.multi_face_landmarks:
                print(f"Skip Item: Caught errors when mediapipe get face_mesh, maybe No face detected in some frames!")
                return None
            else:
                myFaceLandmarks = []
                lms = ret.multi_face_landmarks[0]
                for lm in lms.landmark:
                    myFaceLandmarks.append([lm.x, lm.y, lm.z])
                ldms_normed.append(myFaceLandmarks)
            frame_ids.append(frame_i)
            frame_i += 1
    bs, H, W, _ = frames.shape
    ldms478 = np.array(ldms_normed)
    lm68 = mediapipe_lm478_to_face_alignment_lm68(ldms478, H, W, return_2d=True)
    lm5_lst = [lm68_2_lm5(lm68[i]) for i in range(lm68.shape[0])]
    lm5 = np.stack(lm5_lst)
    return ldms478, lm68, lm5

def process_video_batch(fname_lst, out_name_lst=None):
    frames_lst = []
    with Timer("load_frames", True):
        for (i, res) in multiprocess_run_tqdm(extract_frames_job, fname_lst, num_workers=2, desc="decord is loading frames in the batch videos..."):
            frames_lst.append(res)
    
    lm478s_lst = []
    lm68s_lst = []
    lm5s_lst = []
    with Timer("mediapipe_faceAlign", True):
        for (i, res) in multiprocess_run_tqdm(extract_lms_mediapipe_job, frames_lst, num_workers=2, desc="mediapipe is predicting face mesh in batch videos..."):
            if res is None:
                res = (None, None, None)
            lm478s, lm68s, lm5s = res
            lm478s_lst.append(lm478s)
            lm68s_lst.append(lm68s)
            lm5s_lst.append(lm5s)
        
    processed_cnt_in_this_batch = 0
    with Timer("deep_3drecon_pytorch", True):
        for i, fname in tqdm(enumerate(fname_lst), total=len(fname_lst), desc="extracting 3DMM in the batch videos..."):
            video_rgb = frames_lst[i] # [t, 224,224, 3]
            lm478_arr = lm478s_lst[i]
            lm68_arr = lm68s_lst[i]
            lm5_arr = lm5s_lst[i]
            if lm5_arr is None:
                continue
            num_frames = len(video_rgb)
            batch_size = 32
            iter_times = num_frames // batch_size
            last_bs = num_frames % batch_size

            coeff_lst = []
            for i_iter in range(iter_times):
                start_idx = i_iter * batch_size
                batched_images = video_rgb[start_idx: start_idx + batch_size]
                batched_lm5 = lm5_arr[start_idx: start_idx + batch_size]
                coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
                coeff_lst.append(coeff)
            if last_bs != 0:
                batched_images = video_rgb[-last_bs:]
                batched_lm5 = lm5_arr[-last_bs:]
                coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
                coeff_lst.append(coeff)
            coeff_arr = np.concatenate(coeff_lst,axis=0)
            result_dict = {
                'coeff': coeff_arr.reshape([num_frames, -1]).astype(np.float32),
                'lm478': lm478_arr.reshape([num_frames, 478, 3]).astype(np.float32),
                'lm68': lm68_arr.reshape([num_frames, 68, 2]).astype(np.int16),
                'lm5': lm5_arr.reshape([num_frames, 5, 2]).astype(np.int16),
            }
            np.save(out_name_lst[i], result_dict)
            processed_cnt_in_this_batch +=1
            
    print(f"In this batch {processed_cnt_in_this_batch} files are processed")



def split_wav(mp4_name):
    wav_name = mp4_name[:-4] + '.wav'
    if os.path.exists(wav_name):
        return
    video = VideoFileClip(mp4_name,verbose=False)
    dur = video.duration
    audio = video.audio 
    assert audio is not None
    audio.write_audiofile(wav_name,fps=16000,verbose=False,logger=None)

if __name__ == '__main__':
    ### Process Single Long video for NeRF dataset
    # video_id = 'May'
    # video_fname = f"data/raw/videos/{video_id}.mp4"
    # out_fname = f"data/processed/videos/{video_id}/coeff.npy"
    # process_video(video_fname, out_fname)

    ### Process short video clips for LRS3 dataset
    import random 

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--lrs3_path', type=str, default='/home/yezhenhui/projects/TalkingHead-1KH/datasets/raw/cropped_clips', help='')
    parser.add_argument('--process_id', type=int, default=0, help='')
    parser.add_argument('--total_process', type=int, default=1, help='')
    args = parser.parse_args()

    import os, glob
    lrs3_dir = args.lrs3_path
    out_dir = lrs3_dir.replace("raw/cropped_clips", "processed/coeff")
    os.makedirs(out_dir, exist_ok=True)
    # mp4_name_pattern = os.path.join(lrs3_dir, "*.mp4")
    # mp4_names = glob.glob(mp4_name_pattern)
    with open('/home/yezhenhui/projects/LDMAvatar/clean.txt', 'r') as f:
        txt = f.read()
    mp4_names = txt.split("\n") 
    mp4_names = sorted(mp4_names)
    if args.total_process > 1:
        assert args.process_id <= args.total_process-1
        num_samples_per_process = len(mp4_names) // args.total_process
        if args.process_id == args.total_process-1:
            mp4_names = mp4_names[args.process_id * num_samples_per_process : ]
        else:
            mp4_names = mp4_names[args.process_id * num_samples_per_process : (args.process_id+1) * num_samples_per_process]
    random.seed(111)
    random.shuffle(mp4_names)
    batched_mp4_names_lst = chunk(mp4_names, chunk_size=8)
    for batch_mp4_names in tqdm(batched_mp4_names_lst, desc='[ROOT]: extracting face mesh and 3DMM in batches...'):
        try:
            for mp4_name in batch_mp4_names:
                split_wav(mp4_name)
            out_names = [mp4_name.replace(".mp4", "_coeff_pt.npy").replace("datasets/raw/cropped_clips", "datasets/processed/coeff") for mp4_name in batch_mp4_names]
            process_video_batch(batch_mp4_names, out_names)
            # process_video(mp4_name,out_name=mp4_name.replace(".mp4", "_coeff_pt.npy").replace("datasets/raw/cropped_clips", "datasets/processed/coeff"))
        except Exception as e:
            print(e)
            continue
