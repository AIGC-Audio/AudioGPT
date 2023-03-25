# rnnoise.py, requirements: ffmpeg, sox, rnnoise, python
import os
import subprocess

INSTALL_STR = """
RNNoise library not found. Please install RNNoise (https://github.com/xiph/rnnoise) to $REPO/rnnoise:
sudo apt-get install -y autoconf automake libtool ffmpeg sox
git clone https://github.com/xiph/rnnoise.git
rm -rf rnnoise/.git 
cd rnnoise
./autogen.sh && ./configure && make
cd ..
"""


def rnnoise(filename, out_fn=None, verbose=False, out_sample_rate=22050):
    assert os.path.exists('./rnnoise/examples/rnnoise_demo'), INSTALL_STR
    if out_fn is None:
        out_fn = f"{filename[:-4]}.denoised.wav"
    out_48k_fn = f"{out_fn}.48000.wav"
    tmp0_fn = f"{out_fn}.0.wav"
    tmp1_fn = f"{out_fn}.1.wav"
    tmp2_fn = f"{out_fn}.2.raw"
    tmp3_fn = f"{out_fn}.3.raw"
    if verbose:
        print("Pre-processing audio...")  # wav to pcm raw
    subprocess.check_call(
        f'sox "{filename}" -G -r48000 "{tmp0_fn}"', shell=True, stdin=subprocess.PIPE)  # convert to raw
    subprocess.check_call(
        f'sox -v 0.95 "{tmp0_fn}" "{tmp1_fn}"', shell=True, stdin=subprocess.PIPE)  # convert to raw
    subprocess.check_call(
        f'ffmpeg -y -i "{tmp1_fn}" -loglevel quiet -f s16le -ac 1 -ar 48000 "{tmp2_fn}"',
        shell=True, stdin=subprocess.PIPE)  # convert to raw
    if verbose:
        print("Applying rnnoise algorithm to audio...")  # rnnoise
    subprocess.check_call(
        f'./rnnoise/examples/rnnoise_demo "{tmp2_fn}" "{tmp3_fn}"', shell=True)

    if verbose:
        print("Post-processing audio...")  # pcm raw to wav
    if filename == out_fn:
        subprocess.check_call(f'rm -f "{out_fn}"', shell=True)
    subprocess.check_call(
        f'sox -t raw -r 48000 -b 16 -e signed-integer -c 1 "{tmp3_fn}" "{out_48k_fn}"', shell=True)
    subprocess.check_call(f'sox "{out_48k_fn}" -G -r{out_sample_rate} "{out_fn}"', shell=True)
    subprocess.check_call(f'rm -f "{tmp0_fn}" "{tmp1_fn}" "{tmp2_fn}" "{tmp3_fn}" "{out_48k_fn}"', shell=True)
    if verbose:
        print("Audio-filtering completed!")
