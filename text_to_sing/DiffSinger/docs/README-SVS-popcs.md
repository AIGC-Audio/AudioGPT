## DiffSinger (SVS version)

### 0. Data Acquirement
- See in [apply_form](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md).
- Dataset [preview](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_preview.zip).

### 1. Preparation
#### Data Preparation
a) Download and extract PopCS, then create a link to the dataset folder: `ln -s /xxx/popcs/ data/processed/popcs`

b) Run the following scripts to pack the dataset for training/inference.
```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/popcs_ds_beta6.yaml
# `data/binary/popcs-pmf0` will be generated.
```

#### Vocoder Preparation
We provide the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip) which is specially designed for SVS with NSF mechanism.
Please unzip this file into `checkpoints` before training your acoustic model.

(Update: You can also move [a ckpt with more training steps](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt) into this vocoder directory)

This singing vocoder is trained on ~70 hours singing data, which can be viewed as a universal vocoder. 

### 2. Training Example
First, you need a pre-trained FFT-Singer checkpoint. You can use the [pre-trained model](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_fs2_pmf0_1230.zip), or train FFT-Singer from scratch, run:

```sh
# First, train fft-singer;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_fs2.yaml --exp_name popcs_fs2_pmf0_1230 --reset
# Then, infer fft-singer;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_fs2.yaml --exp_name popcs_fs2_pmf0_1230 --reset --infer 
```

Then, to train DiffSinger, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_ds_beta6_offline_pmf0_1230 --reset
```

Remember to adjust the "fs2_ckpt" parameter in `usr/configs/popcs_ds_beta6_offline.yaml` to fit your path.

### 3. Inference Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_ds_beta6_offline_pmf0_1230 --reset --infer
```

We also provide:
 - the pre-trained model of [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_ds_beta6_offline_pmf0_1230.zip);
 - the pre-trained model of [FFT-Singer](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_fs2_pmf0_1230.zip) for the shallow diffusion mechanism in DiffSinger;

Remember to put the pre-trained models in `checkpoints` directory.

*Note that:* 

- *the original PWG version vocoder in the paper we used has been put into commercial use, so we provide this HifiGAN version vocoder as a substitute.*
- *we assume the ground-truth F0 to be given as the pitch information following [1][2][3]. If you want to conduct experiments on MIDI data, you need an external F0 predictor (like [MIDI-A-version](README-SVS-opencpop-cascade.md)) or a joint prediction with spectrograms(like [MIDI-B-version](README-SVS-opencpop-e2e.md)).*

[1] Adversarially trained multi-singer sequence-to-sequence singing synthesizer. Interspeech 2020.

[2] SEQUENCE-TO-SEQUENCE SINGING SYNTHESIS USING THE FEED-FORWARD TRANSFORMER. ICASSP 2020.

[3] DeepSinger : Singing Voice Synthesis with Data Mined From the Web. KDD 2020.