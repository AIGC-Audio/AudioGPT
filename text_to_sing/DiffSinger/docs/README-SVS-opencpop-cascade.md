# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)

## DiffSinger (MIDI SVS | A version)
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
Please unzip this file into `checkpoints` before training your acoustic model.

(Update: You can also move [a ckpt with more training steps](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt) into this vocoder directory)

This singing vocoder is trained on ~70 hours singing data, which can be viewed as a universal vocoder. 

#### Exp Name Preparation
```bash
export MY_FS_EXP_NAME=0302_opencpop_fs_midi
export MY_DS_EXP_NAME=0303_opencpop_ds58_midi
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
    |--MY_FS_EXP_NAME (optional)
    |--MY_DS_EXP_NAME (optional)
    |--0109_hifigan_bigpopcs_hop128
        |--model_ckpt_steps_1512000.ckpt
        |--config.yaml
```

### 2. Training Example
First, you need a pre-trained FFT-Singer checkpoint. You can use the pre-trained model, or train FFT-Singer from scratch, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/cascade/opencs/aux_rel.yaml --exp_name $MY_FS_EXP_NAME --reset
```

Then, to train DiffSinger, run:

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/cascade/opencs/ds60_rel.yaml --exp_name $MY_DS_EXP_NAME --reset  
```

Remember to adjust the "fs2_ckpt" parameter in `usr/configs/midi/cascade/opencs/ds60_rel.yaml` to fit your path.

### 3. Inference from packed test set
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/cascade/opencs/ds60_rel.yaml --exp_name $MY_DS_EXP_NAME --reset --infer
```

We also provide:
 - the pre-trained model of DiffSinger;
 - the pre-trained model of FFT-Singer;
 
They can be found in [here](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/adjust-receptive-field.zip).

Remember to put the pre-trained models in `checkpoints` directory.

### 4. Inference from raw inputs
```sh
python inference/svs/ds_cascade.py --config usr/configs/midi/cascade/opencs/ds60_rel.yaml --exp_name $MY_DS_EXP_NAME
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

b) in this version of codes, we used the melody frontend ([lyric + MIDI]->[F0+ph_dur]) to predict F0 contour and phoneme duration.

c) generated audio demos can be found in [MY_DS_EXP_NAME](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/adjust-receptive-field.zip).
