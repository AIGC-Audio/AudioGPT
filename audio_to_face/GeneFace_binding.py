import torch

from audio_to_face.inference.postnet_infer import PostnetInfer
from audio_to_face.inference.lm3d_radnerf_infer import LM3d_RADNeRFInfer
from audio_to_face.utils.commons.hparams import set_hparams, hparams

class GeneFaceInfer:
    def __init__(self, device=None):
        self.postnet_inferencer = PostnetInfer(hparams=set_hparams('audio_to_face/checkpoints/May/lm3d_postnet_sync_pitch/config.yaml'),device=device)
        self.radnerf_inferencer = LM3d_RADNeRFInfer(hparams=set_hparams('audio_to_face/checkpoints/May/lm3d_radnerf_torso/config.yaml'),device=device)
    
    def infer_once(self, inp):
        hparams = set_hparams('audio_to_face/checkpoints/May/lm3d_postnet_sync_pitch/config.yaml')
        out_npy_name = self.postnet_inferencer.infer_once(inp)
        hparams = set_hparams('audio_to_face/checkpoints/May/lm3d_radnerf_torso/config.yaml')
        video_name = self.radnerf_inferencer.infer_once(inp)
        return video_name
    
    def example_run(self):
        import os, uuid
        audio_path = 'audio/1f6d012c.wav'
        audio_base_name = os.path.basename(audio_path)[:-4]
        out_video_name = f'video/{str(uuid.uuid4())[0:8]}.mp4'
        inp = {
            'audio_source_name': audio_path,
            'out_npy_name': f'audio_to_face/tmp/{audio_base_name}.npy',
            'cond_name': f'audio_to_face/tmp/{audio_base_name}.npy',
            'out_video_name': out_video_name,
            'tmp_imgs_dir': f'video/tmp_imgs'
        }
        self.infer_once(inp)

if __name__ == '__main__':
    infer = audio_to_faceInfer()
    infer.example_run()