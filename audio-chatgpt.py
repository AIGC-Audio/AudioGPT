import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'NeuralSeq'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio_img'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio_inpaint'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audio_detection'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mono2binaural'))
import gradio as gr
import matplotlib
import librosa
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from diffusers import StableDiffusionPipeline
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import uuid
import soundfile
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from einops import repeat
from ldm.util import instantiate_from_config
from ldm.data.extract_mel_spectrogram import TRANSFORMS_16000
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
from audio_to_text.inference_waveform import AudioCapModel
import whisper
from inference.svs.ds_e2e import DiffSingerE2EInfer
from inference.tts.GenerSpeech import GenerSpeechInfer
from inference.tts.PortaSpeech import TTSInference
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import scipy.io.wavfile as wavfile
import librosa
from audio_infer.utils import config as detection_config
from audio_infer.pytorch.models import PVT
from src.models import BinauralNetwork
from sound_extraction.model.LASSNet import LASSNet
from sound_extraction.utils.stft import STFT
from sound_extraction.utils.wav_io import load_wav, save_wav
from target_sound_detection.src import models as tsd_models
from target_sound_detection.src.models import event_labels
from target_sound_detection.src.utils import median_filter, decode_with_timestamps
from espnet2.bin.svs_inference import SingingGenerate
import clip
import numpy as np
AUDIO_CHATGPT_PREFIX = """Audio ChatGPT
AUdio ChatGPT can not directly read audios, but it has a list of tools to finish different audio synthesis tasks. Each audio will have a file name formed as "audio/xxx.wav". When talking about audios, Audio ChatGPT is very strict to the file name and will never fabricate nonexistent files. 
AUdio ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the audio content and audio file name. It will remember to provide the file name from the last tool observation, if a new audio is generated.
Human may provide Audio ChatGPT with a description. Audio ChatGPT should generate audios according to this description rather than directly imagine from memory or yourself."


TOOLS:
------

Audio ChatGPT  has access to the following tools:"""

AUDIO_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

AUDIO_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if not exists.
You will remember to provide the audio file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}
New input: {input}
Thought: Do I need to use a tool? {agent_scratchpad}"""



def cut_dialogue_history(history_memory, keep_last_n_words = 500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)



def initialize_model(config, ckpt, device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    sampler = DDIMSampler(model)
    return sampler

def initialize_model_inpaint(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print(model.device,device,model.cond_stage_model.device)
    sampler = DDIMSampler(model)
    return sampler

def select_best_audio(prompt,wav_list):
    clap_model = CLAPWrapper('useful_ckpts/CLAP/CLAP_weights_2022.pth','useful_ckpts/CLAP/config.yml',use_cuda=torch.cuda.is_available())
    text_embeddings = clap_model.get_text_embeddings([prompt])
    score_list = []
    for data in wav_list:
        sr,wav = data
        audio_embeddings = clap_model.get_audio_embeddings([(torch.FloatTensor(wav),sr)], resample=True)
        score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False).squeeze().cpu().numpy()
        score_list.append(score)
    max_index = np.array(score_list).argmax()
    print(score_list,max_index)
    return wav_list[max_index]


class T2I:
    def __init__(self, device):
        print("Initializing T2I to %s" % device)
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model, tokenizer=self.text_refine_tokenizer, device=self.device)
        self.pipe.to(device)

    def inference(self, text):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        refined_text = self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        print(f'{text} refined to {refined_text}')
        image = self.pipe(refined_text).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename

class ImageCaptioning:
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions

class T2A:
    def __init__(self, device):
        print("Initializing Make-An-Audio to %s" % device)
        self.device = device
        self.sampler = initialize_model('text_to_audio/Make_An_Audio/configs/text_to_audio/txt2audio_args.yaml', 'text_to_audio/Make_An_Audio/useful_ckpts/ta40multi_epoch=000085.ckpt', device=device)
        self.vocoder = VocoderBigVGAN('text_to_audio/Make_An_Audio/vocoder/logs/bigv16k53w',device=device)

    def txt2audio(self, text, seed = 55, scale = 1.5, ddim_steps = 100, n_samples = 3, W = 624, H = 80):
        SAMPLE_RATE = 16000
        prng = np.random.RandomState(seed)
        start_code = prng.randn(n_samples, self.sampler.model.first_stage_model.embed_dim, H // 8, W // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)
        uc = self.sampler.model.get_learned_conditioning(n_samples * [""])
        c = self.sampler.model.get_learned_conditioning(n_samples * [text])
        shape = [self.sampler.model.first_stage_model.embed_dim, H//8, W//8]  # (z_dim, 80//2^x, 848//2^x)
        samples_ddim, _ = self.sampler.sample(S = ddim_steps,
                                            conditioning = c,
                                            batch_size = n_samples,
                                            shape = shape,
                                            verbose = False,
                                            unconditional_guidance_scale = scale,
                                            unconditional_conditioning = uc,
                                            x_T = start_code)

        x_samples_ddim = self.sampler.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0) # [0, 1]

        wav_list = []
        for idx,spec in enumerate(x_samples_ddim):
            wav = self.vocoder.vocode(spec)
            wav_list.append((SAMPLE_RATE,wav))
        best_wav = select_best_audio(text, wav_list)
        return best_wav

    def inference(self, text, seed = 55, scale = 1.5, ddim_steps = 100, n_samples = 3, W = 624, H = 80):
        melbins,mel_len = 80,624
        with torch.no_grad():
            result = self.txt2audio(
                text = text,
                H = melbins,
                W = mel_len
            )
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, result[1], samplerate = 16000)
        print(f"Processed T2I.run, text: {text}, audio_filename: {audio_filename}")
        return audio_filename

class I2A:
    def __init__(self, device):
        print("Initializing Make-An-Audio-Image to %s" % device)
        self.device = device
        self.sampler = initialize_model('text_to_audio/Make_An_Audio_img/configs/img_to_audio/img2audio_args.yaml', 'text_to_audio/Make_An_Audio_img/useful_ckpts/ta54_epoch=000216.ckpt', device=device)
        self.vocoder = VocoderBigVGAN('text_to_audio/Make_An_Audio_img/vocoder/logs/bigv16k53w',device=device)
    def img2audio(self, image, seed = 55, scale = 3, ddim_steps = 100, W = 624, H = 80):
        SAMPLE_RATE = 16000
        n_samples = 1 # only support 1 sample
        prng = np.random.RandomState(seed)
        start_code = prng.randn(n_samples, self.sampler.model.first_stage_model.embed_dim, H // 8, W // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)
        uc = self.sampler.model.get_learned_conditioning(n_samples * [""])
        #image = Image.fromarray(image)
        image = Image.open(image)
        image = self.sampler.model.cond_stage_model.preprocess(image).unsqueeze(0)
        image_embedding = self.sampler.model.cond_stage_model.forward_img(image)
        c = image_embedding.repeat(n_samples, 1, 1)# shape:[1,77,1280],即还没有变成句子embedding，仍是每个单词的embedding
        shape = [self.sampler.model.first_stage_model.embed_dim, H//8, W//8]  # (z_dim, 80//2^x, 848//2^x)
        samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            x_T=start_code)

        x_samples_ddim = self.sampler.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0) # [0, 1]
        wav_list = []
        for idx,spec in enumerate(x_samples_ddim):
            wav = self.vocoder.vocode(spec)
            wav_list.append((SAMPLE_RATE,wav))
        best_wav = wav_list[0]
        return best_wav
    def inference(self, image, seed = 55, scale = 3, ddim_steps = 100, W = 624, H = 80):
        melbins,mel_len = 80,624
        with torch.no_grad():
            result = self.img2audio(
                image=image,
                H=melbins,
                W=mel_len
            )
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, result[1], samplerate = 16000)
        print(f"Processed I2a.run, image_filename: {image}, audio_filename: {audio_filename}")
        return audio_filename

class TTS:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing PortaSpeech to %s" % device)
        self.device = device
        self.exp_name = 'checkpoints/ps_adv_baseline'
        self.set_model_hparams()
        self.inferencer = TTSInference(self.hp, device)

    def set_model_hparams(self):
        set_hparams(exp_name=self.exp_name, print_hparams=False)
        self.hp = hp

    def inference(self, text):
        self.set_model_hparams()
        inp = {"text": text}
        out = self.inferencer.infer_once(inp)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, out, samplerate=22050)
        return audio_filename

class T2S:
    def __init__(self, device= None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing DiffSinger to %s" % device)
        self.device = device
        self.exp_name = 'checkpoints/0831_opencpop_ds1000'
        self.config= 'NeuralSeq/egs/egs_bases/svs/midi/e2e/opencpop/ds1000.yaml'
        self.set_model_hparams()
        self.pipe = DiffSingerE2EInfer(self.hp, device)
        self.default_inp = {
            'text': '你 说 你 不 SP 懂 为 何 在 这 时 牵 手 AP',
            'notes': 'D#4/Eb4 | D#4/Eb4 | D#4/Eb4 | D#4/Eb4 | rest | D#4/Eb4 | D4 | D4 | D4 | D#4/Eb4 | F4 | D#4/Eb4 | D4 | rest',
            'notes_duration': '0.113740 | 0.329060 | 0.287950 | 0.133480 | 0.150900 | 0.484730 | 0.242010 | 0.180820 | 0.343570 | 0.152050 | 0.266720 | 0.280310 | 0.633300 | 0.444590'
        }

    def set_model_hparams(self):
        set_hparams(config=self.config, exp_name=self.exp_name, print_hparams=False)
        self.hp = hp

    def inference(self, inputs):
        self.set_model_hparams()
        val = inputs.split(",")
        key = ['text', 'notes', 'notes_duration']
        try:
            inp = {k: v for k, v in zip(key, val)}
            wav = self.pipe.infer_once(inp)
        except:
            print('Error occurs. Generate default audio sample.\n')
            inp = self.default_inp
            wav = self.pipe.infer_once(inp)
        #if inputs == '' or len(val) < len(key):
        #    inp = self.default_inp
        #else:
        #    inp = {k:v for k,v in zip(key,val)}
        #wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        wavfile.write(audio_filename, self.hp['audio_sample_rate'], wav.astype(np.int16))
        print(f"Processed T2S.run, audio_filename: {audio_filename}")
        return audio_filename

class t2s_VISinger:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing VISingere to %s" % device)
        tag = 'AQuarterMile/opencpop_visinger1'
        self.model = SingingGenerate.from_pretrained(
            model_tag=str_or_none(tag),
            device=device,
        )
        phn_dur = [[0.        , 0.219     ],
            [0.219     , 0.50599998],
            [0.50599998, 0.71399999],
            [0.71399999, 1.097     ],
            [1.097     , 1.28799999],
            [1.28799999, 1.98300004],
            [1.98300004, 7.10500002],
            [7.10500002, 7.60400009]]
        phn = ['sh', 'i', 'q', 'v', 'n', 'i', 'SP', 'AP']
        score = [[0, 0.50625, 'sh_i', 58, 'sh_i'], [0.50625, 1.09728, 'q_v', 56, 'q_v'], [1.09728, 1.9832100000000001, 'n_i', 53, 'n_i'], [1.9832100000000001, 7.105360000000001, 'SP', 0, 'SP'], [7.105360000000001, 7.604390000000001, 'AP', 0, 'AP']]
        tempo = 70
        tmp = {}
        tmp["label"] = phn_dur, phn
        tmp["score"] = tempo, score
        self.default_inp = tmp

    def inference(self, inputs):
        val = inputs.split(",")
        key = ['text', 'notes', 'notes_duration']
        try: # TODO: input will be update
            inp = {k: v for k, v in zip(key, val)}
            wav = self.model(text=inp)["wav"]
        except:
            print('Error occurs. Generate default audio sample.\n')
            inp = self.default_inp
            wav = self.model(text=inp)["wav"]

        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, wav, samplerate=self.model.fs)
        return audio_filename

class TTS_OOD:
    def __init__(self, device):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing GenerSpeech to %s" % device)
        self.device = device
        self.exp_name = 'checkpoints/GenerSpeech'
        self.config = 'NeuralSeq/modules/GenerSpeech/config/generspeech.yaml'
        self.set_model_hparams()
        self.pipe = GenerSpeechInfer(self.hp, device)

    def set_model_hparams(self):
        set_hparams(config=self.config, exp_name=self.exp_name, print_hparams=False)
        f0_stats_fn = f'{hp["binary_data_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hp['f0_mean'], hp['f0_std'] = np.load(f0_stats_fn)
            hp['f0_mean'] = float(hp['f0_mean'])
            hp['f0_std'] = float(hp['f0_std'])
        hp['emotion_encoder_path'] = 'checkpoints/Emotion_encoder.pt'
        self.hp = hp

    def inference(self, inputs):
        self.set_model_hparams()
        key = ['ref_audio', 'text']
        val = inputs.split(",")
        inp = {k: v for k, v in zip(key, val)}
        wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        wavfile.write(audio_filename, self.hp['audio_sample_rate'], wav.astype(np.int16))
        print(
            f"Processed GenerSpeech.run. Input text:{val[1]}. Input reference audio: {val[0]}. Output Audio_filename: {audio_filename}")
        return audio_filename

class Inpaint:
    def __init__(self, device):
        print("Initializing Make-An-Audio-inpaint to %s" % device)
        self.device = device
        self.sampler = initialize_model_inpaint('text_to_audio/Make_An_Audio_inpaint/configs/inpaint/txt2audio_args.yaml', 'text_to_audio/Make_An_Audio_inpaint/useful_ckpts/inpaint7_epoch00047.ckpt')
        self.vocoder = VocoderBigVGAN('./vocoder/logs/bigv16k53w',device=device)
        self.cmap_transform = matplotlib.cm.viridis
    def make_batch_sd(self, mel, mask, num_samples=1):

        mel = torch.from_numpy(mel)[None,None,...].to(dtype=torch.float32)
        mask = torch.from_numpy(mask)[None,None,...].to(dtype=torch.float32)
        masked_mel = (1 - mask) * mel

        mel = mel * 2 - 1
        mask = mask * 2 - 1
        masked_mel = masked_mel * 2 -1

        batch = {
             "mel": repeat(mel.to(device=self.device), "1 ... -> n ...", n=num_samples),
             "mask": repeat(mask.to(device=self.device), "1 ... -> n ...", n=num_samples),
             "masked_mel": repeat(masked_mel.to(device=self.device), "1 ... -> n ...", n=num_samples),
        }
        return batch
    def gen_mel(self, input_audio_path):
        SAMPLE_RATE = 16000
        sr, ori_wav = wavfile.read(input_audio_path)
        print("gen_mel")
        print(sr,ori_wav.shape,ori_wav)
        ori_wav = ori_wav.astype(np.float32, order='C') / 32768.0 # order='C'是以C语言格式存储，不用管
        if len(ori_wav.shape)==2:# stereo
            ori_wav = librosa.to_mono(ori_wav.T)# gradio load wav shape could be (wav_len,2) but librosa expects (2,wav_len)
        print(sr,ori_wav.shape,ori_wav)
        ori_wav = librosa.resample(ori_wav,orig_sr = sr,target_sr = SAMPLE_RATE)

        mel_len,hop_size = 848,256
        input_len = mel_len * hop_size
        if len(ori_wav) < input_len:
            input_wav = np.pad(ori_wav,(0,mel_len*hop_size),constant_values=0)
        else:
            input_wav = ori_wav[:input_len]

        mel = TRANSFORMS_16000(input_wav)
        return mel
    def gen_mel_audio(self, input_audio):
        SAMPLE_RATE = 16000
        sr,ori_wav = input_audio
        print("gen_mel_audio")
        print(sr,ori_wav.shape,ori_wav)

        ori_wav = ori_wav.astype(np.float32, order='C') / 32768.0 # order='C'是以C语言格式存储，不用管
        if len(ori_wav.shape)==2:# stereo
            ori_wav = librosa.to_mono(ori_wav.T)# gradio load wav shape could be (wav_len,2) but librosa expects (2,wav_len)
        print(sr,ori_wav.shape,ori_wav)
        ori_wav = librosa.resample(ori_wav,orig_sr = sr,target_sr = SAMPLE_RATE)

        mel_len,hop_size = 848,256
        input_len = mel_len * hop_size
        if len(ori_wav) < input_len:
            input_wav = np.pad(ori_wav,(0,mel_len*hop_size),constant_values=0)
        else:
            input_wav = ori_wav[:input_len]
        mel = TRANSFORMS_16000(input_wav)
        return mel
    def show_mel_fn(self, input_audio_path):
        crop_len = 500 # the full mel cannot be showed due to gradio's Image bug when using tool='sketch'
        crop_mel = self.gen_mel(input_audio_path)[:,:crop_len]
        color_mel = self.cmap_transform(crop_mel)
        image = Image.fromarray((color_mel*255).astype(np.uint8))
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        image.save(image_filename)
        return image_filename
    def inpaint(self, batch, seed, ddim_steps, num_samples=1, W=512, H=512):
        model = self.sampler.model

        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, model.first_stage_model.embed_dim, H // 8, W // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)

        c = model.get_first_stage_encoding(model.encode_first_stage(batch["masked_mel"]))
        cc = torch.nn.functional.interpolate(batch["mask"],
                                                size=c.shape[-2:])
        c = torch.cat((c, cc), dim=1) # (b,c+1,h,w) 1 is mask

        shape = (c.shape[1]-1,)+c.shape[2:]
        samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim)


        mel = torch.clamp((batch["mel"]+1.0)/2.0,min=0.0, max=1.0)
        mask = torch.clamp((batch["mask"]+1.0)/2.0,min=0.0, max=1.0)
        predicted_mel = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
        inpainted = (1-mask)*mel+mask*predicted_mel
        inpainted = inpainted.cpu().numpy().squeeze()
        inapint_wav = self.vocoder.vocode(inpainted)

        return inpainted, inapint_wav
    def inference(self, input_audio, mel_and_mask, seed = 55, ddim_steps = 100):
        SAMPLE_RATE = 16000
        torch.set_grad_enabled(False)
        mel_img = Image.open(mel_and_mask['image'])
        mask_img = Image.open(mel_and_mask["mask"])
        show_mel = np.array(mel_img.convert("L"))/255 # 由于展示的mel只展示了一部分，所以需要重新从音频生成mel
        mask = np.array(mask_img.convert("L"))/255
        mel_bins,mel_len = 80,848
        input_mel = self.gen_mel_audio(input_audio)[:,:mel_len]# 由于展示的mel只展示了一部分，所以需要重新从音频生成mel
        mask = np.pad(mask,((0,0),(0,mel_len-mask.shape[1])),mode='constant',constant_values=0)# 将mask填充到原来的mel的大小
        print(mask.shape,input_mel.shape)
        with torch.no_grad():
            batch = self.make_batch_sd(input_mel,mask,num_samples=1)
            inpainted,gen_wav = self.inpaint(
                batch=batch,
                seed=seed,
                ddim_steps=ddim_steps,
                num_samples=1,
                H=mel_bins, W=mel_len
            )
        inpainted = inpainted[:,:show_mel.shape[1]]
        color_mel = self.cmap_transform(inpainted)
        input_len = int(input_audio[1].shape[0] * SAMPLE_RATE / input_audio[0])
        gen_wav = (gen_wav * 32768).astype(np.int16)[:input_len]
        image = Image.fromarray((color_mel*255).astype(np.uint8))
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        image.save(image_filename)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, gen_wav, samplerate = 16000)
        return image_filename, audio_filename
    
class ASR:
    def __init__(self, device):
        print("Initializing Whisper to %s" % device)
        self.device = device
        self.model = whisper.load_model("base", device=device)

    def inference(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text

class A2T:
    def __init__(self, device):
        print("Initializing Audio-To-Text Model to %s" % device)
        self.device = device
        self.model = AudioCapModel("audio_to_text/audiocaps_cntrstv_cnn14rnn_trm")
    def inference(self, audio_path):
        audio = whisper.load_audio(audio_path)
        caption_text = self.model(audio)
        return caption_text[0]

class SoundDetection:
    def __init__(self, device):
        self.device = device
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 14000
        self.model_type = 'PVT'
        self.checkpoint_path = 'audio_detection/audio_infer/useful_ckpts/audio_detection.pth'
        self.classes_num = detection_config.classes_num
        self.labels = detection_config.labels
        self.frames_per_second = self.sample_rate // self.hop_size
        # Model = eval(self.model_type)
        self.model = PVT(sample_rate=self.sample_rate, window_size=self.window_size, 
            hop_size=self.hop_size, mel_bins=self.mel_bins, fmin=self.fmin, fmax=self.fmax, 
            classes_num=self.classes_num)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)

    def inference(self, audio_path):
        # Forward
        (waveform, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        waveform = waveform[None, :]    # (1, audio_length)
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(self.device)
        # Forward
        with torch.no_grad():
            self.model.eval()
            batch_output_dict = self.model(waveform, None)
        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""
        # print('Sound event detection result (time_steps x classes_num): {}'.format(
        #     framewise_output.shape))
        import numpy as np
        import matplotlib.pyplot as plt
        sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
        top_k = 10  # Show top results
        top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
        """(time_steps, top_k)"""
        # Plot result    
        stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=self.window_size, 
            hop_length=self.hop_size, window='hann', center=True)
        frames_num = stft.shape[-1]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, self.frames_per_second))
        axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / self.frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(np.array(self.labels)[sorted_indexes[0 : top_k]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        plt.savefig(image_filename)
        return image_filename

class SoundExtraction:
    def __init__(self, device):
        self.device = device
        self.model_file = 'sound_extraction/useful_ckpts/LASSNet.pt'
        self.stft = STFT()
        import torch.nn as nn
        self.model = nn.DataParallel(LASSNet(device)).to(device)
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def inference(self, inputs):
        #key = ['ref_audio', 'text']
        val = inputs.split(",")
        audio_path = val[0] # audio_path, text
        text = val[1]
        waveform = load_wav(audio_path)
        waveform = torch.tensor(waveform).transpose(1,0)
        mixed_mag, mixed_phase = self.stft.transform(waveform)
        text_query = ['[CLS] ' + text]
        mixed_mag = mixed_mag.transpose(2,1).unsqueeze(0).to(self.device)
        est_mask = self.model(mixed_mag, text_query)
        est_mag = est_mask * mixed_mag  
        est_mag = est_mag.squeeze(1)  
        est_mag = est_mag.permute(0, 2, 1) 
        est_wav = self.stft.inverse(est_mag.cpu().detach(), mixed_phase)
        est_wav = est_wav.squeeze(0).squeeze(0).numpy()  
        #est_path = f'output/est{i}.wav'
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        print('audio_filename ', audio_filename)
        save_wav(est_wav, audio_filename)
        return audio_filename


class Binaural:
    def __init__(self, device):
        self.device = device
        self.model_file = 'mono2binaural/useful_ckpts/m2b/binaural_network.net'
        self.position_file = ['mono2binaural/useful_ckpts/m2b/tx_positions.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions2.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions3.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions4.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions5.txt']
        self.net = BinauralNetwork(view_dim=7,
                      warpnet_layers=4,
                      warpnet_channels=64,
                      )
        self.net.load_from_file(self.model_file)
        self.sr = 48000
    def inference(self, audio_path):
        mono, sr  = librosa.load(path=audio_path, sr=self.sr, mono=True)
        mono = torch.from_numpy(mono)
        mono = mono.unsqueeze(0)
        import numpy as np
        import random
        rand_int = random.randint(0,4)
        view = np.loadtxt(self.position_file[rand_int]).transpose().astype(np.float32)
        view = torch.from_numpy(view)
        if not view.shape[-1] * 400 == mono.shape[-1]:
            mono = mono[:,:(mono.shape[-1]//400)*400] # 
            if view.shape[1]*400 > mono.shape[1]:
                m_a = view.shape[1] - mono.shape[-1]//400 
                rand_st = random.randint(0,m_a)
                view = view[:,m_a:m_a+(mono.shape[-1]//400)] # 
        # binauralize and save output
        self.net.eval().to(self.device)
        mono, view = mono.to(self.device), view.to(self.device)
        chunk_size = 48000  # forward in chunks of 1s
        rec_field =  1000  # add 1000 samples as "safe bet" since warping has undefined rec. field
        rec_field -= rec_field % 400  # make sure rec_field is a multiple of 400 to match audio and view frequencies
        chunks = [
            {
                "mono": mono[:, max(0, i-rec_field):i+chunk_size],
                "view": view[:, max(0, i-rec_field)//400:(i+chunk_size)//400]
            }
            for i in range(0, mono.shape[-1], chunk_size)
        ]
        for i, chunk in enumerate(chunks):
            with torch.no_grad():
                mono = chunk["mono"].unsqueeze(0)
                view = chunk["view"].unsqueeze(0)
                binaural = self.net(mono, view).squeeze(0)
                if i > 0:
                    binaural = binaural[:, -(mono.shape[-1]-rec_field):]
                chunk["binaural"] = binaural
        binaural = torch.cat([chunk["binaural"] for chunk in chunks], dim=-1)
        binaural = torch.clamp(binaural, min=-1, max=1).cpu()
        #binaural = chunked_forwarding(net, mono, view)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        import torchaudio
        torchaudio.save(audio_filename, binaural, sr)
        #soundfile.write(audio_filename, binaural, samplerate = 48000)
        print(f"Processed Binaural.run, audio_filename: {audio_filename}")
        return audio_filename

class TargetSoundDetection:
    def __init__(self, device):
        self.device = device
        self.MEL_ARGS = {
            'n_mels': 64,
            'n_fft': 2048,
            'hop_length': int(22050 * 20 / 1000),
            'win_length': int(22050 * 40 / 1000)
        }
        self.EPS = np.spacing(1)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.event_labels = event_labels
        self.id_to_event =  {i : label for i, label in enumerate(self.event_labels)}
        config = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/run_config.pth', map_location='cpu')
        config_parameters = dict(config)
        config_parameters['tao'] = 0.6
        if 'thres' not in config_parameters.keys():
            config_parameters['thres'] = 0.5
        if 'time_resolution' not in config_parameters.keys():
            config_parameters['time_resolution'] = 125
        model_parameters = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/run_model_7_loss=-0.0724.pt'
                                        , map_location=lambda storage, loc: storage) # load parameter 
        self.model = getattr(tsd_models, config_parameters['model'])(config_parameters,
                    inputdim=64, outputdim=2, time_resolution=config_parameters['time_resolution'], **config_parameters['model_args'])
        self.model.load_state_dict(model_parameters)
        self.model = self.model.to(self.device).eval()
        self.re_embeds = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/text_emb.pth')
        self.ref_mel = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/ref_mel.pth')

    def extract_feature(self, fname):
        import soundfile as sf
        y, sr = sf.read(fname, dtype='float32')
        print('y ', y.shape)
        ti = y.shape[0]/sr
        if y.ndim > 1:
            y = y.mean(1)
        y = librosa.resample(y, sr, 22050)
        lms_feature = np.log(librosa.feature.melspectrogram(y, **self.MEL_ARGS) + self.EPS).T
        return lms_feature,ti
    
    def build_clip(self, text):
        text = clip.tokenize(text).to(self.device) # ["a diagram with dog", "a dog", "a cat"]
        text_features = self.clip_model.encode_text(text)
        return text_features
    
    def cal_similarity(self, target, retrievals):
        ans = []
        #target =torch.from_numpy(target)
        for name in retrievals.keys():
            tmp = retrievals[name]
            #tmp = torch.from_numpy(tmp)
            s = torch.cosine_similarity(target.squeeze(), tmp.squeeze(), dim=0)
            ans.append(s.item())
        return ans.index(max(ans))
    
    def inference(self, text, audio_path):
        target_emb = self.build_clip(text) # torch type
        idx = self.cal_similarity(target_emb, self.re_embeds)
        target_event = self.id_to_event[idx]
        embedding = self.ref_mel[target_event]
        embedding = torch.from_numpy(embedding)
        embedding = embedding.unsqueeze(0).to(self.device).float()
        #print('embedding ', embedding.shape)
        inputs,ti = self.extract_feature(audio_path)
        #print('ti ', ti)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0).to(self.device).float()
        #print('inputs ', inputs.shape)
        decision, decision_up, logit = self.model(inputs, embedding)
        pred = decision_up.detach().cpu().numpy()
        pred = pred[:,:,0]
        frame_num = decision_up.shape[1]
        time_ratio = ti / frame_num
        filtered_pred = median_filter(pred, window_size=1, threshold=0.5)
        #print('filtered_pred ', filtered_pred)
        time_predictions = []
        for index_k in range(filtered_pred.shape[0]):
            decoded_pred = []
            decoded_pred_ = decode_with_timestamps(target_event, filtered_pred[index_k,:])
            if len(decoded_pred_) == 0: # neg deal
                decoded_pred_.append((target_event, 0, 0))
            decoded_pred.append(decoded_pred_)
            for num_batch in range(len(decoded_pred)): # when we test our model,the batch_size is 1
                cur_pred = pred[num_batch]
                # Save each frame output, for later visualization
                label_prediction = decoded_pred[num_batch] # frame predict
                # print(label_prediction)
                for event_label, onset, offset in label_prediction:
                    time_predictions.append({
                        'onset': onset*time_ratio,
                        'offset': offset*time_ratio,})
        ans = ''
        for i,item in enumerate(time_predictions):
            ans = ans + 'segment' + str(i+1) + ' start_time: ' + str(item['onset']) + '  end_time: ' + str(item['offset']) + '\t'
        #print(ans)
        return ans

class ConversationBot:
    def __init__(self):
        print("Initializing AudioChatGPT")
        self.llm = OpenAI(temperature=0)
        self.t2i = T2I(device="cuda:0")
        self.i2t = ImageCaptioning(device="cuda:1")
        self.t2a = T2A(device="cuda:0")
        self.tts = TTS(device="cuda:0")
        self.t2s = T2S(device="cuda:2")
        self.i2a = I2A(device="cuda:1")
        self.a2t = A2T(device="cuda:2")
        self.asr = ASR(device="cuda:1")
        self.inpaint = Inpaint(device="cuda:0")
        self.tts_ood = TTS_OOD(device="cuda:0")
        self.detection = SoundDetection(device="cuda:0")
        self.binaural = Binaural(device="cuda:1")
        self.extraction = SoundExtraction(device="cuda:0")
        self.TSD = TargetSoundDetection(device="cuda:1")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
                 description="useful for when you want to generate an image from a user input text and it saved it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
                             "The input to this tool should be a string, representing the text used to generate image. "),
            Tool(name="Get Photo Description", func=self.i2t.inference,
                 description="useful for when you want to know what is inside the photo. receives image_path as input. "
                             "The input to this tool should be a string, representing the image_path. "),
            Tool(name="Generate Audio From User Input Text", func=self.t2a.inference,
                 description="useful for when you want to generate an audio from a user input text and it saved it to a file."
                             "The input to this tool should be a string, representing the text used to generate audio."),
            Tool(
                name="Generate human speech with style derived from a speech reference and user input text and save it to a file", func= self.tts_ood.inference,
                description="useful for when you want to generate speech samples with styles (e.g., timbre, emotion, and prosody) derived from a reference custom voice."
                            "Like: Generate a speech with style transferred from this voice. The text is xxx., or speak using the voice of this audio. The text is xxx."
                            "The input to this tool should be a comma seperated string of two, representing reference audio path and input text."),
            Tool(name="Generate singing voice From User Input Text, Note and Duration Sequence", func= self.t2s.inference,
                 description="useful for when you want to generate a piece of singing voice (Optional: from User Input Text, Note and Duration Sequence) and save it to a file."
                             "If Like: Generate a piece of singing voice, the input to this tool should be \"\" since there is no User Input Text, Note and Duration Sequence ."
                             "If Like: Generate a piece of singing voice. Text: xxx, Note: xxx, Duration: xxx. "
                             "Or Like: Generate a piece of singing voice. Text is xxx, note is xxx, duration is xxx."
                             "The input to this tool should be a comma seperated string of three, representing text, note and duration sequence since User Input Text, Note and Duration Sequence are all provided."),
            Tool(name="Synthesize Speech Given the User Input Text", func=self.tts.inference,
                 description="useful for when you want to convert a user input text into speech audio it saved it to a file."
                             "The input to this tool should be a string, representing the text used to be converted to speech."),
            Tool(name="Generate Audio From The Image", func=self.i2a.inference,
                 description="useful for when you want to generate an audio based on an image."
                              "The input to this tool should be a string, representing the image_path. "),
            Tool(name="Generate Text From The Audio", func=self.a2t.inference,
                 description="useful for when you want to describe an audio in text, receives audio_path as input."
                             "The input to this tool should be a string, representing the audio_path."),
            Tool(name="Audio Inpainting", func=self.inpaint.show_mel_fn,
                 description="useful for when you want to inpaint a mel spectrum of an audio and predict this audio, this tool will generate a mel spectrum and you can inpaint it, receives audio_path as input, "
                             "The input to this tool should be a string, representing the audio_path."),
            Tool(name="Transcribe speech", func=self.asr.inference,
                 description="useful for when you want to know the text corresponding to a human speech, receives audio_path as input."
                             "The input to this tool should be a string, representing the audio_path."),
            Tool(name="Detect the sound event from the audio", func=self.detection.inference,
                 description="useful for when you want to know what event in the audio and the sound event start or end time, receives audio_path as input. "
                             "The input to this tool should be a string, representing the audio_path. "),
            Tool(name="Sythesize binaural audio from a mono audio input", func=self.binaural.inference,
                 description="useful for when you want to transfer your mono audio into binaural audio, receives audio_path as input. "
                             "The input to this tool should be a string, representing the audio_path. "),
            Tool(name="Extract sound event from mixture audio based on language description", func=self.extraction.inference,
                 description="useful for when you extract target sound from a mixture audio, you can describe the taregt sound by text, receives audio_path and text as input. "
                             "The input to this tool should be a comma seperated string of two, representing mixture audio path and input text."),
            Tool(name="Detect the sound event from the audio based on your descriptions", func=self.TSD.inference,
                 description="useful for when you want to know the when happens the target sound event in th audio. You can use language descriptions to instruct the model. receives text description and audio_path as input. "
                             "The input to this tool should be a string, representing the answer. ")]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': AUDIO_CHATGPT_PREFIX, 'format_instructions': AUDIO_CHATGPT_FORMAT_INSTRUCTIONS, 'suffix': AUDIO_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        if res['intermediate_steps'] == []:
            print("======>Current memory:\n %s" % self.agent.memory)
            response = res['output']
            state = state + [(text, response)]
            print("Outputs:", state)
            return state, state, gr.Audio.update(visible=False), gr.Image.update(visible=False), gr.Button.update(visible=False)
        else:
            tool = res['intermediate_steps'][0][0].tool
            if tool == "Generate Image From User Input Text" or tool == "Generate Text From The Audio" or tool == "Transcribe speech":
                print("======>Current memory:\n %s" % self.agent.memory)
                response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
                state = state + [(text, response)]
                print("Outputs:", state)
                return state, state, gr.Audio.update(visible=False), gr.Image.update(visible=False), gr.Button.update(visible=False)
            elif tool == "Audio Inpainting":
                audio_filename = res['intermediate_steps'][0][0].tool_input
                image_filename = res['intermediate_steps'][0][1]
               # self.is_visible(True)
                print("======>Current memory:\n %s" % self.agent.memory)
                print(res)
                response = res['output']
                state = state + [(text, response)]
                print("Outputs:", state)
                return state, state, gr.Audio.update(value=audio_filename,visible=True), gr.Image.update(value=image_filename,visible=True), gr.Button.update(visible=True)
            print("======>Current memory:\n %s" % self.agent.memory)
            response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
            audio_filename = res['intermediate_steps'][0][1]
            state = state + [(text, response)]
            print("Outputs:", state)
            return state, state, gr.Audio.update(value=audio_filename,visible=True), gr.Image.update(visible=False), gr.Button.update(visible=False)

    def run_image_or_audio(self, file, state, txt):
        file_type = file.name[-3:]
        if file_type == "wav":
            print("===============Running run_audio =============")
            print("Inputs:", file, state)
            print("======>Previous memory:\n %s" % self.agent.memory)
            audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
            audio_load = whisper.load_audio(file.name)
            soundfile.write(audio_filename, audio_load, samplerate = 16000)
            description = self.a2t.inference(audio_filename)
            Human_prompt = "\nHuman: provide an audio named {}. The description is: {}. This information helps you to understand this audio, but you should use tools to finish following tasks, " \
                           "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(audio_filename, description)
            AI_prompt = "Received.  "
            self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
            print("======>Current memory:\n %s" % self.agent.memory)
            #state = state + [(f"<audio src=audio_filename controls=controls></audio>*{audio_filename}*", AI_prompt)]
            state = state + [(f"*{audio_filename}*", AI_prompt)]
            print("Outputs:", state)
            return state, state, txt + ' ' + audio_filename + ' ', gr.Audio.update(value=audio_filename,visible=True)
        else:
            print("===============Running run_image =============")
            print("Inputs:", file, state)
            print("======>Previous memory:\n %s" % self.agent.memory)
            image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
            print("======>Auto Resize Image...")
            img = Image.open(file.name)
            width, height = img.size
            ratio = min(512 / width, 512 / height)
            width_new, height_new = (round(width * ratio), round(height * ratio))
            img = img.resize((width_new, height_new))
            img = img.convert('RGB')
            img.save(image_filename, "PNG")
            print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
            description = self.i2t.inference(image_filename)
            Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
                           "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(image_filename, description)
            AI_prompt = "Received.  "
            self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
            print("======>Current memory:\n %s" % self.agent.memory)
            state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
            print("Outputs:", state)
            return state, state, txt + ' ' + image_filename + ' ', gr.Audio.update(visible=False)

    def inpainting(self, state, audio_filename, image_filename):
        print("===============Running inpainting =============")
        print("Inputs:", state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        inpaint = Inpaint(device="cuda:0")
        new_image_filename, new_audio_filename = inpaint.inference(audio_filename, image_filename)
        AI_prompt = "Here are the predict audio and the mel spectrum." + f"*{new_audio_filename}*" + f"![](/file={new_image_filename})*{new_image_filename}*"
        self.agent.memory.buffer = self.agent.memory.buffer + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"Audio Inpainting", AI_prompt)]
        print("Outputs:", state)
        return state, state, gr.Image.update(visible=False), gr.Audio.update(value=new_audio_filename, visible=True), gr.Button.update(visible=False)
    def clear_audio(self):
        return gr.Audio.update(value=None, visible=False)
    def clear_image(self):
        return gr.Image.update(value=None, visible=False)
    def clear_button(self):
        return gr.Button.update(visible=False)


if __name__ == '__main__':
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        with gr.Row():
            gr.Markdown("## Audio ChatGPT")
        chatbot = gr.Chatbot(elem_id="chatbot", label="Audio ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear️")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image","audio"])
        with gr.Column():
            outaudio = gr.Audio(visible=False)
        with gr.Row():
            with gr.Column():
                show_mel = gr.Image(type="filepath",tool='sketch',visible=False)
                run_button = gr.Button("Predict Masked Place",visible=False)


        txt.submit(bot.run_text, [txt, state], [chatbot, state, outaudio, show_mel, run_button])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image_or_audio, [btn, state, txt], [chatbot, state, txt, outaudio])
        run_button.click(bot.inpainting, [state, outaudio, show_mel], [chatbot, state, show_mel, outaudio, run_button])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        clear.click(lambda:None, None, txt)
        clear.click(bot.clear_button, None, run_button)
        clear.click(bot.clear_image, None, show_mel)
        clear.click(bot.clear_audio, None, outaudio)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)