import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_sing/DiffSinger'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio_img'))
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import os
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import uuid
import soundfile
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import cv2
import einops
from pytorch_lightning import seed_everything
import random
from ldm.util import instantiate_from_config
from ldm.data.extract_mel_spectrogram import TRANSFORMS_16000
from pathlib import Path
from vocoder.hifigan.modules import VocoderHifigan
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
from inference.svs.ds_e2e import DiffSingerE2EInfer
import whisper
from text_to_speech.TTS_binding import TTSInference

AUDIO_CHATGPT_PREFIX = """Audio ChatGPT


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

#temp_audio_filename = "audio/c00d9240.wav"


def cut_dialogue_history(history_memory, keep_last_n_words = 500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    return os.path.join(head, new_file_name)


def initialize_model(config, ckpt, device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
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

class MaskFormer:
    def __init__(self, device):
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt",).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(image.size)


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
        self.sampler = initialize_model('configs/text-to-audio/txt2audio_args.yaml', 'useful_ckpts/ta40multi_epoch=000085.ckpt', device=device) 
        self.vocoder = VocoderHifigan('vocoder/logs/hifi_0127',device=device)

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
        self.inferencer = TTSInference(device)
    
    def inference(self, text):
        global temp_audio_filename
        inp = {"text": text}
        out = self.inferencer.infer_once(inp)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        temp_audio_filename = audio_filename
        soundfile.write(audio_filename, out, samplerate = 22050)
        return audio_filename
    
class T2S:
    def __init__(self, device= None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing DiffSinger to %s" % device)
        self.device = device
        exp_name = 'text_to_sing/DiffSinger/checkpoints/0831_opencpop_ds1000'
        exp_name = 'checkpoints/0831_opencpop_ds1000'
        config= 'text_to_sing/DiffSinger/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml'
        from utils.hparams import set_hparams
        from utils.hparams import hparams as hp
        set_hparams(config= config,exp_name=exp_name, print_hparams=False)
        self.hp = hp
        self.pipe = DiffSingerE2EInfer(self.hp)
    def inference(self, inputs):
        key = ['text', 'notes', 'notes_duration']
        val = inputs.split(",")
        print(val)
        inp = {k:v for k,v in zip(key,val)}
        print(inp)
        wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, wav.astype(np.int16), self.hp['audio_sample_rate'])
        print(f"Processed T2S.run, text: {val[0]}, notes: {val[1]}, notes duration: {val[2]}, audio_filename: {audio_filename}")
        return audio_filename

# need to debug
class Inpaint:
    def __init__(self, device):
        print("Initializing Make-An-Audio-inpaint to %s" % device)
        self.device = device
        self.sampler = initialize_model('text_to_audio/Make_An_Audio_inpaint/configs/inpaint/txt2audio_args.yaml', 'text_to_audio/Make_An_Audio_inpaint/useful_ckpts/inpaint7_epoch00047.ckpt')
        self.vocoder = VocoderBigVGAN('./vocoder/logs/bigv16k53w',device=device)
    def make_batch_sd(mel, mask, num_samples=1):

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
    def gen_mel(input_audio):
        sr,ori_wav = input_audio
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
    def show_mel_fn(input_audio):
        crop_len = 500 # the full mel cannot be showed due to gradio's Image bug when using tool='sketch'
        crop_mel = self.gen_mel(input_audio)[:,:crop_len]
        color_mel = cmap_transform(crop_mel)
        return Image.fromarray((color_mel*255).astype(np.uint8))
    def inpaint(batch, seed, ddim_steps, num_samples=1, W=512, H=512):
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

    
        mask = batch["mask"]# [-1,1]
        mel = torch.clamp((batch["mel"]+1.0)/2.0,min=0.0, max=1.0)
        mask = torch.clamp((batch["mask"]+1.0)/2.0,min=0.0, max=1.0)
        predicted_mel = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
        inpainted = (1-mask)*mel+mask*predicted_mel
        inpainted = inpainted.cpu().numpy().squeeze()
        inapint_wav = self.vocoder.vocode(inpainted)

        return inpainted, inapint_wav
    def predict(input_audio,mel_and_mask,ddim_steps,seed):
        show_mel = np.array(mel_and_mask['image'].convert("L"))/255 # 由于展示的mel只展示了一部分，所以需要重新从音频生成mel
        mask = np.array(mel_and_mask["mask"].convert("L"))/255

        mel_bins,mel_len = 80,848

        input_mel = self.gen_mel(input_audio)[:,:mel_len]# 由于展示的mel只展示了一部分，所以需要重新从音频生成mel
        mask = np.pad(mask,((0,0),(0,mel_len-mask.shape[1])),mode='constant',constant_values=0)# 将mask填充到原来的mel的大小
        print(mask.shape,input_mel.shape)
        with torch.no_grad():
            batch = make_batch_sd(input_mel,mask,device,num_samples=1)
            inpainted,gen_wav = self.inpaint(
                batch=batch,
                seed=seed,
                ddim_steps=ddim_steps,
                num_samples=1,
                H=mel_bins, W=mel_len
            )
        inpainted = inpainted[:,:show_mel.shape[1]]
        color_mel = cmap_transform(inpainted)
        input_len = int(input_audio[1].shape[0] * SAMPLE_RATE / input_audio[0])
        gen_wav = (gen_wav * 32768).astype(np.int16)[:input_len]
        return Image.fromarray((color_mel*255).astype(np.uint8)),(SAMPLE_RATE,gen_wav)
    
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

class ConversationBot:
    def __init__(self):
        print("Initializing AudioChatGPT")
        self.llm = OpenAI(temperature=0)
        self.t2i = T2I(device="cuda:0")
        self.i2t = ImageCaptioning(device="cuda:1")
        self.t2a = T2A(device="cuda:0")
        self.t2s = T2S(device="cuda:2")
        self.i2a = I2A(device="cuda:1")
        self.asr = ASR(device="cuda:1")
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
            Tool(name="Generate singing voice From User Input Text", func=self.t2s.inference,
                 description="useful for when you want to generate a piece of singing voice from its description."
                             "The input to this tool should be a comma seperated string of three, representing the text sequence and its corresponding note and duration sequence."),
            Tool(name="Generate Audio From The Image", func=self.i2a.inference,
                 description="useful for when you want to generate an audio based on an image."
                             "The input to this tool should be a string, representing the image_path. "),
            Tool(name="Get Audio Transcription", func=self.asr.inference,
                 description="useful for when you want to know the text content corresponding to this audio, receives audio_path as input."
                             "The input to this tool should be a string, representing the audio_path.")
                            ]
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
        tool = res['intermediate_steps'][0][0].tool
        if tool == "Generate Image From User Input Text":
            print("======>Current memory:\n %s" % self.agent.memory)
            response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
            state = state + [(text, response)]
            print("Outputs:", state)
            return state, state, None
        print("======>Current memory:\n %s" % self.agent.memory)
        audio_filename = res['intermediate_steps'][0][1]
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        #response = res['output'] + f"<audio src=audio_filename controls=controls></audio>"
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state, audio_filename

    def run_image_or_audio(self, file, state, txt):
        file_type = file.name[-3:]
        if file_type == "wav":
            print("===============Running run_audio =============")
            print("Inputs:", file, state)
            print("======>Previous memory:\n %s" % self.agent.memory)
            audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
            print("======>Auto Resize Audio...")
            audio_load = whisper.load_audio(file.name)
            soundfile.write(audio_filename, audio_load, samplerate = 16000)
            description = self.asr.inference(audio_filename)
            Human_prompt = "\nHuman: provide an audio named {}. The description is: {}. This information helps you to understand this audio, but you should use tools to finish following tasks, " \
                           "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(audio_filename, description)
            AI_prompt = "Received.  "
            self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
            #state = state + [(f"<audio src=audio_filename controls=controls></audio>*{audio_filename}*", AI_prompt)]
            state = state + [(f"*{audio_filename}*", AI_prompt)]
            print("Outputs:", state)
            return state, state, txt + ' ' + audio_filename + ' ', audio_filename
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
            return state, state, txt + ' ' + image_filename + ' ', None

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
            outaudio = gr.Audio()
        txt.submit(bot.run_text, [txt, state], [chatbot, state, outaudio])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image_or_audio, [btn, state, txt], [chatbot, state, txt, outaudio])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
