# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## DiffSinger (SVS)

### PART1. [Run DiffSinger on PopCS](README-SVS-popcs.md)
In PART1, we only focus on spectrum modeling (acoustic model) and assume the ground-truth (GT) F0 to be given as the pitch information following these papers [1][2][3]. If you want to conduct experiments with F0 prediction, please move to PART2.

Thus, the pipeline of this part can be summarized as:

```
[lyrics] -> [linguistic representation] (Frontend)
[linguistic representation] + [GT F0] + [GT phoneme duration] -> [mel-spectrogram]  (Acoustic model)
[mel-spectrogram] + [GT F0] -> [waveform] (Vocoder)
```


[1] Adversarially trained multi-singer sequence-to-sequence singing synthesizer. Interspeech 2020.

[2] SEQUENCE-TO-SEQUENCE SINGING SYNTHESIS USING THE FEED-FORWARD TRANSFORMER. ICASSP 2020.

[3] DeepSinger : Singing Voice Synthesis with Data Mined From the Web. KDD 2020.

Click here for detailed instructions: [link](README-SVS-popcs.md).


### PART2. [Run DiffSinger on Opencpop](README-SVS-opencpop-cascade.md)
Thanks [Opencpop team](https://wenet.org.cn/opencpop/) for releasing their SVS dataset with MIDI label, **Jan.20, 2022** (after we published our paper).

Since there are elaborately annotated MIDI labels, we are able to supplement the pipeline in PART 1 by adding a naive melody frontend.

#### 2.A
Thus, the pipeline of [2.A](README-SVS-opencpop-cascade.md) can be summarized as:

```
[lyrics] + [MIDI] -> [linguistic representation (with MIDI information)] + [predicted F0] + [predicted phoneme duration] (Melody frontend)
[linguistic representation] + [predicted F0] + [predicted phoneme duration] -> [mel-spectrogram]  (Acoustic model)
[mel-spectrogram] + [predicted F0] -> [waveform] (Vocoder)
```

Click here for detailed instructions: [link](README-SVS-opencpop-cascade.md).

#### 2.B
In 2.1, we find that if we predict F0 explicitly in the melody frontend, there will be many bad cases of uv/v prediction. Then, we abandon the explicit prediction of the F0 curve in the melody frontend and make a joint prediction with spectrograms.

Thus, the pipeline of [2.B](README-SVS-opencpop-e2e.md) can be summarized as:
```
[lyrics] + [MIDI] -> [linguistic representation] + [predicted phoneme duration] (Melody frontend)
[linguistic representation (with MIDI information)] + [predicted phoneme duration] -> [mel-spectrogram]  (Acoustic model)
[mel-spectrogram] -> [predicted F0]  (Pitch extractor)
[mel-spectrogram] + [predicted F0] -> [waveform] (Vocoder)
```

Click here for detailed instructions: [link](README-SVS-opencpop-e2e.md).

### FAQ
Q1: Why do I need F0 in Vocoders?

A1: See vocoder parts in HiFiSinger, DiffSinger or SingGAN. This is a common practice now.

Q2: Why not run MIDI version SVS on PopCS dataset? or Why not release MIDI labels for PopCS dataset?

A2: Our laboratory has no funds to label PopCS dataset. But there are funds for labeling other singing dataset, which is coming soon.

Q3: Why " 'HifiGAN' object has no attribute 'model' "?

A3: Please put the pretrained vocoders in your `checkpoints` dictionary.

Q4: How to check whether I use GT information or predicted information during inference from packed test set?

A4: Please see codes [here](https://github.com/MoonInTheRiver/DiffSinger/blob/55e2f46068af6e69940a9f8f02d306c24a940cab/tasks/tts/fs2.py#L343).

...