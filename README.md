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

### Speech
|            Task            |   Supported Foundation Models   | Status |
|:--------------------------:|:-------------------------------:|:------:|
|       Text-to-Speech       | [FastSpeech](), [SyntaSpeech](), [VITS]() |  Yes (WIP)   |
|       Style Transfer       |         [GenerSpeech]()         |  Yes   |
|     Speech Recognition     |           [whisper](), [Conformer]()           |  Yes   |
|     Speech Enhancement     |          [ConvTasNet]()         |  Yes (WIP)   |
|     Speech Separation      |          [TF-GridNet]()         |  Yes (WIP)   |
|     Speech Translation     |          [Multi-decoder]()      |  WIP   |
|      Mono-to-Binaural      |          [NeuralWarp]()         |  Yes   |

### Sing

|           Task            |   Supported Foundation Models   | Status |
|:-------------------------:|:-------------------------------:|:------:|
|       Text-to-Sing        |         [DiffSinger](), [VISinger]()          |  Yes (WIP)   |

### Audio
|          Task          | Supported Foundation Models | Status |
|:----------------------:|:---------------------------:|:------:|
|     Text-to-Audio      |      [Make-An-Audio]()      |  Yes   |
|    Audio Inpainting    |      [Make-An-Audio]()      |  Yes   |
|     Image-to-Audio     |      [Make-An-Audio]()      |  Yes   |
|    Sound Detection     |    [Audio-transformer]()    | Yes    |
| Target Sound Detection |    [TSDNet]()    |  Yes   |
|    Sound Extraction    |    [LASSNet]()    |  Yes   |


### Talking Head

|           Task            |   Supported Foundation Models   |   Status   |
|:-------------------------:|:-------------------------------:|:----------:|
|  Talking Head Synthesis   |          [GeneFace]()           | Yes (WIP)  |


## Acknowledgement
We appreciate the open source of the following projects:

[ESPNet](https://github.com/espnet/espnet) &#8194;
[NATSpeech](https://github.com/NATSpeech/NATSpeech) &#8194;
[Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) &#8194;
[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194;

