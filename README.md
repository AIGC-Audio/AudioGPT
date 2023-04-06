# AudioGPT

**AudioGPT** connects ChatGPT and a series of Audio Foundation Models to enable **sending** and **receiving** speech, sing, audio, and talking head during chatting.

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/AIGC-Audio/AudioGPT">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Spaces">
</a>

## Capabilities

Up-to-date link: https://93868c7fa583f4b5.gradio.app

Here we list the capability of AudioGPT at this time. More supported models and tasks are comming soon. For prompt examples, refer to [asset](assets/README.md).

### Speech
|            Task            |   Supported Foundation Models   | Status |
|:--------------------------:|:-------------------------------:|:------:|
|       Text-to-Speech       | [FastSpeech](), [SyntaSpeech](), [VITS]() |  Yes (WIP)   |
|       Style Transfer       |         [GenerSpeech]()         |  Yes   |
|     Speech Recognition     |           [whisper](), [Conformer]()           |  Yes   |
|     Speech Enhancement     |          [ConvTasNet]()         |  WIP   |
|     Speech Separation      |          [TF-GridNet]()         |  WIP   |
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

## Internal Version Updates
4.6 Support Sound Extraction/Detection\
4.3 Support huggingface demo space\
4.1 Support Audio inpainting and clean codes\
3.27 Support Style Transfer/Talking head Synthesis\
3.23 Support Text-to-Sing\
3.21 Support Image-to-Audio\
3.19 Support Speech Recognition\
3.17 Support Text-to-Audio

## Todo
- [x] clean text to sing/speech code
- [ ] merge talking head synthesis into main
- [x] change audio/video log output
- [x] support huggingface space

## Acknowledgement
We appreciate the open source of the following projects:

[Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) &#8194;
[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194;
