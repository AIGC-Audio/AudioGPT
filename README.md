# AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2304.12995)
[![GitHub Stars](https://img.shields.io/github/stars/AIGC-Audio/AudioGPT?style=social)](https://github.com/AIGC-Audio/AudioGPT)
![visitors](https://visitor-badge.glitch.me/badge?page_id=AIGC-Audio.AudioGPT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/AudioGPT)


We provide our implementation and pretrained models as open source in this repository.


## Get Started

Please refer to [run.md](run.md)


## Capabilities

Here we list the capability of AudioGPT at this time. More supported models and tasks are coming soon. For prompt examples, refer to [asset](assets/README.md).

Currently not every model has repository.
### Speech
|            Task            |   Supported Foundation Models   | Status |
|:--------------------------:|:-------------------------------:|:------:|
|       Text-to-Speech       | [FastSpeech](https://github.com/ming024/FastSpeech2), [SyntaSpeech](https://github.com/yerfor/SyntaSpeech), [VITS](https://github.com/jaywalnut310/vits) |  Yes (WIP)   |
|       Style Transfer       |         [GenerSpeech](https://github.com/Rongjiehuang/GenerSpeech)         |  Yes   |
|     Speech Recognition     |           [whisper](https://github.com/openai/whisper), [Conformer](https://github.com/sooftware/conformer)           |  Yes   |
|     Speech Enhancement     |          [ConvTasNet]()         |  Yes (WIP)   |
|     Speech Separation      |          [TF-GridNet](https://arxiv.org/pdf/2211.12433.pdf)         |  Yes (WIP)   |
|     Speech Translation     |          [Multi-decoder](https://arxiv.org/pdf/2109.12804.pdf)      |  WIP   |
|      Mono-to-Binaural      |          [NeuralWarp](https://github.com/fdarmon/NeuralWarp)         |  Yes   |

### Sing

|           Task            |   Supported Foundation Models   | Status |
|:-------------------------:|:-------------------------------:|:------:|
|       Text-to-Sing        |         [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger), [VISinger](https://github.com/jerryuhoo/VISinger)          |  Yes (WIP)   |

### Audio
|          Task          | Supported Foundation Models | Status |
|:----------------------:|:---------------------------:|:------:|
|     Text-to-Audio      |      [Make-An-Audio]()      |  Yes   |
|    Audio Inpainting    |      [Make-An-Audio]()      |  Yes   |
|     Image-to-Audio     |      [Make-An-Audio]()      |  Yes   |
|    Sound Detection     |    [Audio-transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)    | Yes    |
| Target Sound Detection |    [TSDNet](https://github.com/gy65896/TSDNet)    |  Yes   |
|    Sound Extraction    |    [LASSNet](https://github.com/liuxubo717/LASS)    |  Yes   |


### Talking Head

|           Task            |   Supported Foundation Models   |   Status   |
|:-------------------------:|:-------------------------------:|:----------:|
|  Talking Head Synthesis   |          [GeneFace](https://github.com/yerfor/GeneFace)           | Yes (WIP)  |


## Acknowledgement
We appreciate the open source of the following projects:

[ESPNet](https://github.com/espnet/espnet) &#8194;
[NATSpeech](https://github.com/NATSpeech/NATSpeech) &#8194;
[Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) &#8194;
[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194;

