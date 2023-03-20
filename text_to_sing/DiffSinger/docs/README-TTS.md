# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 
## DiffSpeech (TTS)
### 1. Preparation

#### Data Preparation
a) Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then create a link to the dataset folder: `ln -s /xxx/LJSpeech-1.1/ data/raw/`

b) Download and Unzip the [ground-truth duration](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/mfa_outputs.tar) extracted by [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz):  `tar -xvf mfa_outputs.tar; mv mfa_outputs data/processed/ljspeech/`

c) Run the following scripts to pack the dataset for training/inference.

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config configs/tts/lj/fs2.yaml

# `data/binary/ljspeech` will be generated.
```

#### Vocoder Preparation
We provide the pre-trained model of [HifiGAN](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0414_hifi_lj_1.zip) vocoder.
Please unzip this file into `checkpoints` before training your acoustic model.

### 2. Training Example

First, you need a pre-trained FastSpeech2 checkpoint. You can use the [pre-trained model](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/fs2_lj_1.zip), or train FastSpeech2 from scratch, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config configs/tts/lj/fs2.yaml --exp_name fs2_lj_1 --reset
```
Then, to train DiffSpeech, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_ds_beta6_1213 --reset
```

Remember to adjust the "fs2_ckpt" parameter in `usr/configs/lj_ds_beta6.yaml` to fit your path.

### 3. Inference Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_ds_beta6_1213 --reset --infer
```

We also provide:
 - the pre-trained model of [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/lj_ds_beta6_1213.zip);
 - the individual pre-trained model of [FastSpeech 2](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/fs2_lj_1.zip) for the shallow diffusion mechanism in DiffSpeech;
 
Remember to put the pre-trained models in `checkpoints` directory.

## Mel Visualization
Along vertical axis, DiffSpeech: [0-80]; FastSpeech2: [80-160].

<table style="width:100%">
  <tr>
    <th>DiffSpeech vs. FastSpeech 2</th>
  </tr>
  <tr>
    <td><img src="resources/diffspeech-fs2.png" alt="DiffSpeech-vs-FastSpeech2" height="250"></td>
  </tr>
  <tr>
    <td><img src="resources/diffspeech-fs2-1.png" alt="DiffSpeech-vs-FastSpeech2" height="250"></td>
  </tr>
  <tr>
    <td><img src="resources/diffspeech-fs2-2.png" alt="DiffSpeech-vs-FastSpeech2" height="250"></td>
  </tr>
</table>