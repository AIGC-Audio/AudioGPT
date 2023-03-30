import os
import sys
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import importlib
import tqdm
import logging
import copy
import re
import random

from audio_to_face.utils.commons.ddp_utils import DDP
from audio_to_face.utils.commons.hparams import hparams, set_hparams
from audio_to_face.utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
from audio_to_face.utils.commons.euler2rot import euler_trans_2_c2w, c2w_to_euler_trans
from audio_to_face.utils.commons.tensor_utils import move_to_cpu, move_to_cuda, convert_to_tensor

from audio_to_face.tasks.nerfs.dataset_utils import NeRFDataset
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def smooth_camera_path(poses, kernel_size=7):
    # smooth the camera trajectory (i.e., translation)...
    # poses: [N, 4, 4], numpy array
    N = poses.shape[0]
    K = kernel_size // 2
    
    trans = poses[:, :3, 3].copy() # [N, 3]
    rots = poses[:, :3, :3].copy() # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        try:
            poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()
        except:
            if i == 0:
                poses[i, :3, :3] = rots[i]
            else:
                poses[i, :3, :3] = poses[i-1, :3, :3]
    return poses


class BaseNeRFInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.infer_max_length = hparams.get('infer_max_length', 500000) # default render 10 seconds long
        self.device = device
        self.dataset_cls = NeRFDataset # the dataset only provides head pose 
        self.dataset = self.dataset_cls('trainval')

        assert hparams['task_cls'] != ''
        pkg = ".".join(hparams["task_cls"].split(".")[:-1])
        cls_name = hparams["task_cls"].split(".")[-1]
        self.task_cls = getattr(importlib.import_module(pkg), cls_name)

        self.all_gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
        self.num_gpus = len(self.all_gpu_ids)
        self.on_gpu = self.num_gpus > 0
        self.root_gpu = 0
        logging.info(f'GPU available: {torch.cuda.is_available()}, GPU used: {self.all_gpu_ids}')
        self.use_ddp = self.num_gpus > 1
        self.proc_rank = 0

    def build_nerf_task(self):
        task = self.task_cls()
        task.build_model()
        task.eval()
        load_ckpt(task.model, hparams['work_dir'], 'model')
        ckpt, _ = get_last_checkpoint(hparams['work_dir'])
        task.global_step = ckpt['global_step']
        return task

    def _forward_nerf_task_single_process(self, batches):
        tmp_imgs_dir = self.inp['tmp_imgs_dir']
        os.makedirs(tmp_imgs_dir, exist_ok=True)
        H, W = batches[0]['H'], batches[0]['W']
        H = int(hparams['infer_scale_factor']*H)
        W = int(hparams['infer_scale_factor']*W)
        idx_batch_lst = [(idx, batch) for idx,batch in enumerate(batches)]

        print(f"The tmp imge dir is {tmp_imgs_dir}.")
        with torch.no_grad():
            for (idx, batch) in tqdm.tqdm(idx_batch_lst, total=len(idx_batch_lst),
                                desc=f"NeRF is rendering frames..."):
                torch.cuda.empty_cache()
                if 'cuda' in self.device:
                    batch = move_to_cuda(batch)
                model_out = self.nerf_task.run_model(batch, infer=True)
                pred_rgb = model_out['rgb_map'] * 255
                pred_img = pred_rgb.view([H, W, 3]).cpu().numpy().astype(np.uint8)
                out_name = os.path.join(tmp_imgs_dir, format(idx, '05d')+".png")
                bgr_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_name, bgr_img)
                batches[idx] = move_to_cpu(batch)
                for k in list(batch.keys()):
                    del batch[k]
                torch.cuda.empty_cache()
        return tmp_imgs_dir

    def init_ddp_connection(self, proc_rank, world_size):
        root_node = '127.0.0.1'
        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)
    
    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]
            number = re.sub('[^0-9]', '', number)
            root_node = name + number
        return root_node

    def configure_ddp(self, task):
        task = DDP(task, device_ids=[self.root_gpu], find_unused_parameters=True)
        random.seed(self.hparams['seed'])
        np.random.seed(self.hparams['seed'])
        return task

    def _forward_nerf_task_ddp(self, gpu_idx, batches, hparams_):
        hparams.update(hparams_) # the global hparams dict in the subprocess is empty, so inplace-update it!
        self.proc_rank = gpu_idx
        self.init_ddp_connection(self.proc_rank, self.num_gpus)
        # if dist.get_rank() != 0:
        #     sys.stdout = open(os.devnull, "w")
        #     sys.stderr = open(os.devnull, "w")
        tmp_imgs_dir = self.inp['tmp_imgs_dir']
        os.makedirs(tmp_imgs_dir, exist_ok=True)
        torch.cuda.set_device(gpu_idx)
        self.root_gpu = gpu_idx
        self.nerf_task = self.build_nerf_task()
        self.nerf_task.eval()
        self.nerf_task.cuda()
        self.nerf_task = self.configure_ddp(self.nerf_task)
        dist.barrier()
        nerf_task = self.nerf_task.module
        self.dataset = self.dataset_cls('train')

        idx_batch_lst = [(idx, batch) for idx,batch in enumerate(batches)]
        num_batchs_per_gpu = len(batches) // self.num_gpus
        if self.proc_rank != self.num_gpus-1:
            idx_batch_lst = idx_batch_lst[self.proc_rank*num_batchs_per_gpu:(self.proc_rank+1)*num_batchs_per_gpu]
        else:
            idx_batch_lst = idx_batch_lst[self.proc_rank*num_batchs_per_gpu:]
        
        H, W = batches[0]['H'], batches[0]['W']
        H = int(hparams['infer_scale_factor']*H)
        W = int(hparams['infer_scale_factor']*W)
        with torch.no_grad():
            if dist.get_rank() == 0:
                print(f"The tmp imge dir is {tmp_imgs_dir}.")
            for (idx, batch) in tqdm.tqdm(idx_batch_lst, total=len(idx_batch_lst),
                                desc=f"Process {self.proc_rank} : NeRF is rendering frames..."):
                torch.cuda.empty_cache()
                if self.device == 'cuda':
                    batch = move_to_cuda(batch, self.root_gpu)
                model_out = nerf_task.run_model(batch, infer=True)
                pred_rgb = model_out['rgb_map'] * 255
                pred_img = pred_rgb.view([H, W, 3]).cpu().numpy().astype(np.uint8)
                out_name = os.path.join(tmp_imgs_dir, format(idx, '05d')+".png")
                bgr_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_name, bgr_img)
                batches[idx] = move_to_cpu(batch)
                for k in list(batch.keys()):
                    del batch[k]
                torch.cuda.empty_cache()
        dist.barrier()
        return tmp_imgs_dir

    def forward_system(self, batches):
        if self.use_ddp:
            del self.dataset
            torch.multiprocessing.set_sharing_strategy('file_system')
            batches = copy.deepcopy(batches)
            mp.spawn(self._forward_nerf_task_ddp, nprocs=self.num_gpus, args=[batches, copy.deepcopy(hparams)])
            img_dir = self.inp['tmp_imgs_dir']
        else:
            self.nerf_task = self.build_nerf_task()
            self.nerf_task.eval()
            self.nerf_task.to(self.device)
            img_dir = self._forward_nerf_task_single_process(batches)
        return img_dir

    def get_cond_from_input(self, inp):
        """
        get the conditon features of NeRF
        """
        raise NotImplementedError

    def get_pose_from_ds(self, samples):
        """
        process the item into torch.tensor batch
        """
        if self.use_pred_pose:
            print(f"The head pose mode is: pred")
            c2w_arr = np.load(self.inp['c2w_name'])[0] # [T, 3, 3]
            print(f"Loaded head pose from {self.inp['c2w_name']}.")
            assert len(samples) - len(c2w_arr) < 5
            if len(samples) > len(c2w_arr):
                samples = samples[:len(c2w_arr)]
            if len(samples) < len(c2w_arr):
                c2w_arr = c2w_arr[:len(samples)]
        else:
            print(f"The head pose mode is: gt")
            
        for idx, sample in enumerate(samples):
            if idx >= len(self.dataset.samples) and not self.use_pred_pose:
                # since we use GT head pose from the dataset, the pred_samples cannot be longer than the GT samples
                del samples[idx:]
                break
            sample['H'] = self.dataset.H
            sample['W'] = self.dataset.W
            sample['focal'] = self.dataset.focal
            sample['cx'] = self.dataset.cx
            sample['cy'] = self.dataset.cy
            sample['near'] = hparams['near']
            sample['far'] = hparams['far']
            sample['bg_img'] = self.dataset.bg_img

            if self.use_pred_pose:
                sample['c2w'] = torch.from_numpy(c2w_arr[idx])
            else:
                sample['c2w'] = self.dataset.samples[idx]['c2w'][:3]
            sample['c2w_t0'] = self.dataset.samples[0]['c2w'][:3]

            sample['t'] = torch.tensor([0,]).float()
            euler, trans = c2w_to_euler_trans(sample['c2w'])
            euler_t0, trans_t0 = c2w_to_euler_trans(sample['c2w_t0'])
            sample['euler'] = torch.tensor(np.ascontiguousarray(euler)).float()
            sample['trans'] = torch.tensor(np.ascontiguousarray(trans)).float()
            sample['euler_t0'] = torch.tensor(np.ascontiguousarray(euler_t0)).float()
            sample['trans_t0'] = torch.tensor(np.ascontiguousarray(trans_t0)).float()
            
        if hparams.get("infer_smo_head_pose", True) is True:
            c2w_arr = torch.stack([s['c2w'] for s in samples]).numpy()
            smo_c2w_arr = smooth_camera_path(c2w_arr)
            for i, sample in enumerate(samples):
                sample['c2w'] = convert_to_tensor(smo_c2w_arr[i])
                euler, trans = c2w_to_euler_trans(sample['c2w'])
                sample['euler'] = convert_to_tensor(np.ascontiguousarray(euler))
                sample['trans'] = convert_to_tensor(np.ascontiguousarray(trans))
        return samples

    def postprocess_output(self, output):
        tmp_imgs_dir = self.inp['tmp_imgs_dir']
        out_video_name = self.inp['out_video_name']
        self.save_mp4(tmp_imgs_dir, self.wav16k_name, out_video_name) 
        return out_video_name

    def infer_once(self, inp):
        self.inp = inp
        self.use_pred_pose = True if self.inp.get('c2w_name','') != '' else False
        samples = self.get_cond_from_input(inp)
        batches = self.get_pose_from_ds(samples)
        image_dir = self.forward_system(batches)
        if self.proc_rank == 0:
            out_name = self.postprocess_output(image_dir)
            print(f"The synthesized video is saved at {out_name}")
        return out_name
    
    @classmethod
    def example_run(cls, inp=None):
        from audio_to_face.utils.commons.hparams import set_hparams
        from audio_to_face.utils.commons.hparams import hparams as hp
        set_hparams()
        inp_tmp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'out_dir': 'infer_out',
            'out_video_name': 'infer_out/zozo.mp4'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp
        if hparams.get("infer_cond_name", '') != '':
            inp['cond_name'] = hparams['infer_cond_name']
        if hparams.get("infer_audio_source_name", '') != '':
            inp['audio_source_name'] = hparams['infer_audio_source_name'] 
        if hparams.get("infer_out_video_name", '') != '':
            inp['out_video_name'] = hparams['infer_out_video_name']
        if hparams.get("infer_c2w_name", '') != '':
            inp['c2w_name'] = hparams['infer_c2w_name']
        out_dir = os.path.dirname(inp['out_video_name'])
        video_name = os.path.basename(inp['out_video_name'])[:-4]
        tmp_imgs_dir = os.path.join(out_dir, "tmp_imgs", video_name)
        inp['tmp_imgs_dir'] = tmp_imgs_dir

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(tmp_imgs_dir, exist_ok=True)
        infer_ins = cls(hp)
        infer_ins.infer_once(inp)

    ##############
    # IO-related
    ##############
    @classmethod
    def save_mp4(self, img_dir, wav_name, out_name):
        # os.system(f"ffmpeg -i {img_dir}/%5d.png -i {wav_name} -shortest -pix_fmt yuv420p -b:v 2000k -r 25 -strict -2 -y {out_name}")
        os.system(f"ffmpeg -i {img_dir}/%5d.png -i {wav_name} -shortest -c:v libx264 -pix_fmt yuv420p -b:v 2000k -r 25 -strict -2 -y {out_name}")
        # os.system(f"ffmpeg -i {img_dir}/%5d.png -i {wav_name} -shortest -v quiet -c:v libx264 -pix_fmt yuv420p -b:v 2000k -r 25 -strict -2 -y {out_name}")

    def save_wav16k(self, inp):
        source_name = inp['audio_source_name']
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert source_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = source_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {source_name} -v quiet -f wav -ar 16000 {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Saved 16khz wav file to {wav16k_name}.")
