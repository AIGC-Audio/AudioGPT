# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

Substantial update: We 1) **abandon** the explicit prediction of the F0 curve; 2) increase the receptive field of the denoiser; 3) make the linguistic encoder more robust.
**By doing so, 1) the synthesized recordings are more natural in terms of pitch; 2) the pipeline is simpler.**

简而言之，把F0曲线的动态性交给生成式模型去捕捉，而不再是以前那样用MSE约束对数域F0。

## DiffSinger (MIDI SVS | B version)
### 0. Data Acquirement
For Opencpop dataset: Please strictly follow the instructions of [Opencpop](https://wenet.org.cn/opencpop/). We have no right to give you the access to Opencpop.

The pipeline below is designed for Opencpop dataset:

### 1. Preparation

#### Data Preparation
a) Download and extract Opencpop, then create a link to the dataset folder: `ln -s /xxx/opencpop data/raw/`

b) Run the following scripts to pack the dataset for training/inference.

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/midi/cascade/opencs/aux_rel.yaml

# `data/binary/opencpop-midi-dp` will be generated.
```

#### Vocoder Preparation
We provide the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip) which is specially designed for SVS with NSF mechanism.

Also, please unzip pre-trained vocoder and [this pendant for vocoder](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0102_xiaoma_pe.zip) into `checkpoints` before training your acoustic model.

(Update: You can also move [a ckpt with more training steps](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt) into this vocoder directory)

This singing vocoder is trained on ~70 hours singing data, which can be viewed as a universal vocoder. 

#### Exp Name Preparation
```bash
export MY_DS_EXP_NAME=0228_opencpop_ds100_rel
```

```
.
|--data
    |--raw
        |--opencpop
            |--segments
                |--transcriptions.txt
                |--wavs
|--checkpoints
    |--MY_DS_EXP_NAME (optional)
    |--0109_hifigan_bigpopcs_hop128 (vocoder)
        |--model_ckpt_steps_1512000.ckpt
        |--config.yaml
```

### 2. Training Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME --reset  
```

### 3. Inference from packed test set
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME --reset --infer
```

We also provide:
 - the pre-trained model of DiffSinger;
 
They can be found in [here](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0228_opencpop_ds100_rel.zip).

Remember to put the pre-trained models in `checkpoints` directory.

### 4. Inference from raw inputs
```sh
python inference/svs/ds_e2e.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME
```
Raw inputs:
```
inp = {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
        'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
        'input_type': 'word'
    }  # user input: Chinese characters
or,
inp = {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
        'note_seq': 'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 F#4/Gb4 F#4/Gb4 F#4/Gb4 C#4/Db4 C#4/Db4 C#4/Db4 rest C#4/Db4 C#4/Db4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 F4 F4 C#4/Db4 C#4/Db4',
        'note_dur_seq': '0.407140 0.407140 0.376190 0.376190 0.242180 0.242180 0.509550 0.509550 0.183420 0.315400 0.315400 0.235020 0.361660 0.361660 0.223070 0.377270 0.377270 0.340550 0.340550 0.299620 0.299620 0.344510 0.344510 0.283770 0.283770 0.323390 0.323390 0.360340 0.360340',
        'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
        'input_type': 'phoneme'
    }  # input like Opencpop dataset.
```

### 5. Some issues.
a) the HifiGAN-Singing is trained on our [vocoder dataset](https://dl.acm.org/doi/abs/10.1145/3474085.3475437) and the training set of [PopCS](https://arxiv.org/abs/2105.02446). Opencpop is the out-of-domain dataset (unseen speaker). This may cause the deterioration of audio quality, and we are considering fine-tuning this vocoder on the training set of Opencpop.

b) in this version of codes, we used the melody frontend ([lyric + MIDI]->[ph_dur]) to predict phoneme duration. F0 curve is implicitly predicted together with mel-spectrogram.

c) example [generated audio](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0221/DS/).
More generated audio demos can be found in [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0228_opencpop_ds100_rel.zip).
