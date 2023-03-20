# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 | [English README](../README.md)

本仓库包含了我们的AAAI-2022 [论文](https://arxiv.org/abs/2105.02446)中提出的DiffSpeech (用于语音合成) 与 DiffSinger (用于歌声合成) 的官方Pytorch实现。

<table style="width:100%">
  <tr>
    <th>DiffSinger/DiffSpeech训练阶段</th>
    <th>DiffSinger/DiffSpeech推理阶段</th>
  </tr>
  <tr>
    <td><img src="resources/model_a.png" alt="Training" height="300"></td>
    <td><img src="resources/model_b.png" alt="Inference" height="300"></td>
  </tr>
</table>

:tada: :tada: :tada: **一些重要更新**:
- Mar.2, 2022: [MIDI-新版](README-SVS-opencpop-e2e.md): 重大更新 :sparkles:
 - Mar.1, 2022: [NeuralSVB](https://github.com/MoonInTheRiver/NeuralSVB), 为了歌声美化任务的代码，开源了 :sparkles:  :sparkles:  :sparkles: .
 - Feb.13, 2022: [NATSpeech](https://github.com/NATSpeech/NATSpeech), 一个升级后的代码框架, 包含了DiffSpeech和我们NeurIPS-2021的工作[PortaSpeech](https://openreview.net/forum?id=xmJsuh8xlq) 已经开源! :sparkles: :sparkles: :sparkles:. 
 - Jan.29, 2022: 支持了[MIDI-旧版](README-SVS-opencpop-cascade.md) 版本的歌声合成系统.
 - Jan.13, 2022: 支持了歌声合成系统, 开源了PopCS数据集.
 - Dec.19, 2021: 支持了语音合成系统. [HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 
:rocket: **新闻**: 
 - Feb.24, 2022: 我们的新工作`NeuralSVB` 被 ACL-2022 接收 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277). [音频演示](https://neuralsvb.github.io).
 - Dec.01, 2021: DiffSinger被AAAI-2022接收.
 - Sep.29, 2021: 我们的新工作`PortaSpeech: Portable and High-Quality Generative Text-to-Speech` 被NeurIPS-2021接收 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2109.15166) .
 - May.06, 2021: 我们把这篇DiffSinger提交到了公开论文网站: Arxiv [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446).

## 安装依赖
```sh
conda create -n your_env_name python=3.8
source activate your_env_name 
pip install -r requirements_2080.txt   (GPU 2080Ti, CUDA 10.2)
or pip install -r requirements_3090.txt   (GPU 3090, CUDA 11.4)
```

## DiffSpeech (语音合成的版本)
### 1. 准备工作

#### 数据准备
a) 下载并解压 [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), 创建软链接: `ln -s /xxx/LJSpeech-1.1/ data/raw/`

b) 下载并解压 [我们用MFA预处理好的对齐](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/mfa_outputs.tar):  `tar -xvf mfa_outputs.tar; mv mfa_outputs data/processed/ljspeech/`

c) 按照如下脚本给数据集打包，打包后的二进制文件用于后续的训练和推理.

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config configs/tts/lj/fs2.yaml

# `data/binary/ljspeech` will be generated.
```

#### 声码器准备
我们提供了[HifiGAN](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0414_hifi_lj_1.zip)声码器的预训练模型.
请在训练声学模型前，先把声码器文件解压到`checkpoints`里。

### 2. 训练样例

首先你需要一个预训练好的FastSpeech2存档点. 你可以用[我们预训练好的模型](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/fs2_lj_1.zip), 或者跑下面这个指令从零开始训练FastSpeech2:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config configs/tts/lj/fs2.yaml --exp_name fs2_lj_1 --reset
```
然后为了训练DiffSpeech, 运行:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_ds_beta6_1213 --reset
```

记得针对你的路径修改`usr/configs/lj_ds_beta6.yaml`里"fs2_ckpt"这个参数.

### 3. 推理样例

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_ds_beta6_1213 --reset --infer
```

我们也提供了:
 - [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/lj_ds_beta6_1213.zip)的预训练模型;
 - [FastSpeech 2](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/fs2_lj_1.zip)的预训练模型, 这是为了DiffSpeech里的浅扩散机制;
 
记得把预训练模型放在 `checkpoints` 目录.

## DiffSinger (歌声合成的版本)

### 0. 数据获取
- 见 [申请表](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md).
- 数据集 [预览](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_preview.zip).

### 1. Preparation
#### 数据准备
a) 下载并解压PopCS, 创建软链接: `ln -s /xxx/popcs/ data/processed/popcs`

b) 按照如下脚本给数据集打包，打包后的二进制文件用于后续的训练和推理.
```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/popcs_ds_beta6.yaml
# `data/binary/popcs-pmf0` 会生成出来.
```

#### 声码器准备
我们提供了[HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip)的预训练模型, 它专门为了歌声合成系统设计, 采用了NSF的技术。
请在训练声学模型前，先把声码器文件解压到`checkpoints`里。

(更新: 你也可以将我们提供的[训练更多步数的存档点](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt)放到声码器的文件夹里)

这个声码器是在大约70小时的较大数据集上训练的, 可以被认为是一个通用声码器。

### 2. 训练样例
首先你需要一个预训练好的FFT-Singer. 你可以用[我们预训练好的模型](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_fs2_pmf0_1230.zip), 或者用如下脚本从零训练FFT-Singer:

```sh
# First, train fft-singer;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_fs2.yaml --exp_name popcs_fs2_pmf0_1230 --reset
# Then, infer fft-singer;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_fs2.yaml --exp_name popcs_fs2_pmf0_1230 --reset --infer 
```

然后, 为了训练DiffSinger, 运行:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_ds_beta6_offline_pmf0_1230 --reset
```

记得针对你的路径修改`usr/configs/popcs_ds_beta6_offline.yaml`里"fs2_ckpt"这个参数.

### 3. 推理样例
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_ds_beta6_offline_pmf0_1230 --reset --infer
```

我们也提供了:
 - [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_ds_beta6_offline_pmf0_1230.zip)的预训练模型;
 - [FFT-Singer](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_fs2_pmf0_1230.zip)的预训练模型, 这是为了DiffSinger里的浅扩散机制;

记得把预训练模型放在 `checkpoints` 目录.

*请注意：*

-*我们原始论文中的PWG版本声码器已投入商业使用，因此我们提供此HifiGAN版本声码器作为替代品。*

-*我们这篇论文假设提供真实的F0来进行实验，如[1][2][3]等前作所做的那样，重点在频谱建模上，而非F0曲线的预测。如果你想对MIDI数据进行实验，从MIDI和歌词预测F0曲线（显式或隐式），请查看文档[MIDI-old-version](README-SVS-opencpop-cascade.md) 或 [MIDI-new-version](README-SVS-opencpop-e2e.md)。目前已经支持的MIDI数据集有: Opencpop*

[1] Adversarially trained multi-singer sequence-to-sequence singing synthesizer. Interspeech 2020.

[2] SEQUENCE-TO-SEQUENCE SINGING SYNTHESIS USING THE FEED-FORWARD TRANSFORMER. ICASSP 2020.

[3] DeepSinger : Singing Voice Synthesis with Data Mined From the Web. KDD 2020.

## Tensorboard
```sh
tensorboard --logdir_spec exp_name
```
<table style="width:100%">
  <tr>
    <td><img src="resources/tfb.png" alt="Tensorboard" height="250"></td>
  </tr>
</table>

## Mel 可视化
沿着纵轴, DiffSpeech: [0-80]; FastSpeech2: [80-160].

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

## Audio Demos
音频样本可以看我们的[样例页](https://diffsinger.github.io/).

我们也放了部分由DiffSpeech+HifiGAN (标记为[P]) 和 GTmel+HifiGAN (标记为[G]) 生成的测试集音频样例在：[resources/demos_1213](../resources/demos_1213). 

(对应这个预训练参数：[DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/lj_ds_beta6_1213.zip))

---
:rocket: :rocket: :rocket: **更新:**

新生成的歌声样例在：[resources/demos_0112](../resources/demos_0112).

## Citation
如果本仓库对你的研究和工作有用，请引用以下论文：

    @article{liu2021diffsinger,
      title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
      author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
      journal={arXiv preprint arXiv:2105.02446},
      volume={2},
      year={2021}}


## 鸣谢
我们的代码基于如下仓库:
* [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)
* [DiffWave](https://github.com/lmnt-com/diffwave)