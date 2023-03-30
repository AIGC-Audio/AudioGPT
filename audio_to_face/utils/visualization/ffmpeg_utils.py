import os

def imgs_to_video(img_dir, video_path, audio_path=None, verbose=False):
    cmd = f"ffmpeg -i {img_dir}/%5d.png "
    if audio_path is not None:
        cmd += f"-i {audio_path} "
        cmd += "-strict -2 "
    cmd += "-c:v libx264 -pix_fmt yuv420p -b:v 2000k -y "
    if verbose is False:
        cmd += " -v quiet "
    cmd += f"{video_path} "

    os.system(cmd)


if __name__ == '__main__':
    imgs_to_video('infer_out/tmp_imgs', 'infer_out/tmp_imgs/out.mp4', 'data/raw/val_wavs/zozo.wav')
    imgs_to_video('infer_out/tmp_imgs', 'infer_out/tmp_imgs/out2.mp4', 'data/raw/val_wavs/zozo.wav')