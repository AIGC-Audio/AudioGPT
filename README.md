# AudioGPT

**AudioGPT** connects ChatGPT and a series of Audio Foundation Models to enable **sending** and **receiving** speech, sing, audio, and talking head during chatting.


## Capabilities

Up-to-date link: https://eac422a9e2289d6b.gradio.app/

Here we list the capability of AudioGPT at this time. More supported models and tasks are comming soon. For prompt examples, refer to [asset](assets/README.md).

### Speech
|           Task            |   Supported Foundation Models   | Status |
|:-------------------------:|:-------------------------------:|:------:|
|      Text-to-Speech       | [FastSpeech](), [SyntaSpeech](), [VITS]() |  Yes (WIP)   |
|      Style Transfer       |         [GenerSpeech]()         |  Yes   |
|    Speech Recognition     |           [whisper](), [Conformer]()           |  Yes   |
|    Speech Enhancement     |          [ConvTasNet]()         |  WIP   |
|    Speech Separation      |          [TF-GridNet]()         |  WIP   |
|    Speech Translation     |          [Multi-decoder]()      |  WIP   |
|  Mono-to-Binaural Speech  |          [NeuralWarp]()         |  Yes   |

### Sing

|           Task            |   Supported Foundation Models   | Status |
|:-------------------------:|:-------------------------------:|:------:|
|       Text-to-Sing        |         [DiffSinger](), [VISinger]()          |  Yes (WIP)   |

### Audio
|           Task            |   Supported Foundation Models   | Status |
|:-------------------------:|:-------------------------------:|:------:|
|       Text-to-Audio       |        [Make-An-Audio]()        |  Yes   |
|     Audio Inpainting      |        [Make-An-Audio]()        |  WIP   |
|      Image-to-Audio       |        [Make-An-Audio]()        |  Yes   |
|     sound detection       |        [Audio-transformer]()    |  Yes   |


### Talking Head

|           Task            |   Supported Foundation Models   | Status |
|:-------------------------:|:-------------------------------:|:------:|
|  Talking Head Synthesis   |          [GeneFace]()           |  WIP   |

## Internal Version Updates
3.27 Support Style Transfer/Talking head Synthesis\
3.23 Support Text-to-Sing\
3.21 Support Image-to-Sing\
3.19 Support Speech Recognition\
3.17 Support Text-to-Audio

## Todo
- [ ] clean text to sing/speech code
- [ ] import Espnet models for speech tasks
- [ ] merge talking head synthesis into main
- [ ] change audio/video log output
- [ ] support huggingface space

## Acknowledgement
We appreciate the open source of the following projects:

[Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) &#8194;
[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194;
