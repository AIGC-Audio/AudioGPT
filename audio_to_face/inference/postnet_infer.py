import os
import torch
import librosa
import numpy as np
import importlib
import tqdm

from audio_to_face.utils.commons.tensor_utils import move_to_cuda
from audio_to_face.utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
from audio_to_face.utils.commons.hparams import hparams, set_hparams


class PostnetInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.infer_max_length = hparams.get('infer_max_length', 500000)
        self.device = device
        self.postnet_task = self.build_postnet_task()
        self.postnet_task.eval()
        self.postnet_task.to(self.device)
        
    def build_postnet_task(self):
        assert hparams['task_cls'] != ''
        pkg = ".".join(hparams["task_cls"].split(".")[:-1])
        cls_name = hparams["task_cls"].split(".")[-1]
        task_cls = getattr(importlib.import_module(pkg), cls_name)
        task = task_cls()
        task.build_model()
        task.eval()
        steps = hparams.get('infer_ckpt_steps', 12000)
        load_ckpt(task.model, hparams['work_dir'], 'model', steps=steps)
        load_ckpt(task.audio2motion_task, hparams['work_dir'], 'audio2motion_task', steps=steps)
        load_ckpt(task.syncnet_task, hparams['work_dir'], 'syncnet_task', steps=steps)
        task.global_step = steps
        return task
    

    def infer_once(self, inp):  
        self.inp = inp
        samples = self.get_cond_from_input(inp)
        out_name = self.forward_system(samples, inp)
        print(f"The predicted and refined 3D landmark sequence is saved at {out_name}")
        return out_name
    
    def get_cond_from_input(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a list that contains the condition feature of NeRF
        """
        self.save_wav16k(inp)
        from audio_to_face.data_gen.process_lrs3.process_audio_hubert import get_hubert_from_16k_wav
        hubert = get_hubert_from_16k_wav(self.wav16k_name).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0)))

        from audio_to_face.data_gen.process_lrs3.process_audio_mel_f0 import extract_mel_from_fname,extract_f0_from_wav_and_mel
        wav, mel = extract_mel_from_fname(self.wav16k_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        f0 = f0.reshape([-1,1])
        if f0.shape[0] > len(hubert):
            f0 = f0[:len(hubert)]
        else:
            num_to_pad = len(hubert) - len(f0)
            f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))
        f0 = f0.squeeze(-1)
        
        t_x = hubert.shape[0]
        x_mask = torch.ones([1, t_x]).float()
        y_mask = torch.ones([1, t_x//2]).float()
        sample = {
            'hubert': torch.from_numpy(hubert).float().unsqueeze(0),
            'f0': torch.from_numpy(f0).float().unsqueeze(0),
            'x_mask': x_mask,
            'y_mask': y_mask,
            }
        return [sample]

    def forward_system(self, batches, inp):
        out_dir = self._forward_postnet_task(batches, inp)
        return out_dir

    def _forward_postnet_task(self, batches, inp):
        with torch.no_grad():
            pred_lst = []            
            for idx, batch in tqdm.tqdm(enumerate(batches), total=len(batches),
                                desc=f"Now VAE is predicting the action into {inp['out_npy_name']}"):
                if self.device == 'cuda':
                    batch = move_to_cuda(batch)
                model_out = self.postnet_task.run_model(batch, infer=True, temperature=1.)
                pred = model_out['refine_lm3d'].squeeze().cpu().numpy()
                pred_lst.append(pred)
        np.save(inp['out_npy_name'], pred_lst)
        return inp['out_npy_name']

    @classmethod
    def example_run(cls, inp=None):
        inp_tmp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'out_npy_name': 'infer_out/lrs3/0.npy'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp
        if hparams.get("infer_audio_source_name", '') != '':
            inp['audio_source_name'] = hparams['infer_audio_source_name'] 
        if hparams.get("infer_out_npy_name", '') != '':
            inp['out_npy_name'] = hparams['infer_out_npy_name']
        out_dir = os.path.dirname(inp['out_npy_name'])

        os.makedirs(out_dir, exist_ok=True)
        infer_ins = cls(hparams)
        infer_ins.infer_once(inp)

    ##############
    # IO-related
    ##############
    def save_wav16k(self, inp):
        source_name = inp['audio_source_name']
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert source_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = source_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {source_name} -f wav -ar 16000 {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {source_name} to {wav16k_name}.")

if __name__ == '__main__':
    set_hparams()
    inp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'out_npy_name': 'infer_out/May/pred_lm3d/zozo.npy',
            }
    PostnetInfer.example_run(inp)