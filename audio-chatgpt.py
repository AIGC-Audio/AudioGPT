import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
from pathlib import Path
from vocoder.hifigan.modules import VocoderHifigan
from ldm.models.diffusion.ddim import DDIMSampler
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
import whisper

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
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}
New input: {input}
Thought: Do I need to use a tool? {agent_scratchpad}"""

SAMPLE_RATE = 16000
temp_audio_filename = "audio/c00d9240.wav"
# model = whisper.load_model("base")

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

clap_model = CLAPWrapper('useful_ckpts/CLAP/CLAP_weights_2022.pth','useful_ckpts/CLAP/config.yml',use_cuda=torch.cuda.is_available())

def select_best_audio(prompt,wav_list):
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


class T2A:
    def __init__(self, device):
        print("Initializing Make-An-Audio to %s" % device)
        self.device = device
        self.sampler = initialize_model('configs/text-to-audio/txt2audio_args.yaml', 'useful_ckpts/ta40multi_epoch=000085.ckpt', device=device) 
        self.vocoder = VocoderHifigan('vocoder/logs/hifi_0127',device=device)

    def txt2audio(self, text, seed = 55, scale = 1.5, ddim_steps = 100, n_samples = 3, W = 624, H = 80):
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
        global temp_audio_filename
        melbins,mel_len = 80,624
        with torch.no_grad():
            result = self.txt2audio(
                text = text,
                H = melbins,
                W = mel_len
            )
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        temp_audio_filename = audio_filename
        soundfile.write(audio_filename, result[1], samplerate = 16000)
        print(f"Processed T2I.run, text: {text}, audio_filename: {audio_filename}")
        return audio_filename

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
        #print(f"Detected language: {max(probs, key=probs.get)}")
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text

class ConversationBot:
    def __init__(self):
        print("Initializing AudioChatGPT")
        self.llm = OpenAI(temperature=0)
        self.t2i = T2I(device="cuda:2")
        self.t2a = T2A(device="cuda:2")
        self.asr = ASR(device="cuda:2")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
                 description="useful for when you want to generate an image from a user input text and it saved it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
                             "The input to this tool should be a string, representing the text used to generate image. "),
            Tool(name="Generate Audio From User Input Text", func=self.t2a.inference,
                 description="useful for when you want to generate an audio from a user input text and it saved it to a file."
                             "The input to this tool should be a string, representing the text used to generate audio."),
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
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state, temp_audio_filename

    def run_audio(self, audio, state, txt):
        print("===============Running run_audio =============")
        print("Inputs:", audio, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        print("======>Auto Resize Audio...")
        audio_load = whisper.load_audio(audio.name)
        soundfile.write(audio_filename, audio_load, samplerate = 16000)
        global temp_audio_filename
        temp_audio_filename = audio_filename
        description = self.asr.inference(audio_filename)
        Human_prompt = "\nHuman: provide an audio named {}. The description is: {}. This information helps you to understand this audio, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(audio_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"*{audio_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + ' ' + audio_filename + ' ', temp_audio_filename

    # def run_image(self, image, state, txt):
    #     print("===============Running run_image =============")
    #     print("Inputs:", image, state)
    #     print("======>Previous memory:\n %s" % self.agent.memory)
    #     image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
    #     print("======>Auto Resize Image...")
    #     img = Image.open(image.name)
    #     width, height = img.size
    #     ratio = min(512 / width, 512 / height)
    #     width_new, height_new = (round(width * ratio), round(height * ratio))
    #     img = img.resize((width_new, height_new))
    #     img = img.convert('RGB')
    #     img.save(image_filename, "PNG")
    #     print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
    #     description = self.i2t.inference(image_filename)
    #     Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
    #                    "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(image_filename, description)
    #     AI_prompt = "Received.  "
    #     self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
    #     print("======>Current memory:\n %s" % self.agent.memory)
    #     state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
    #     print("Outputs:", state)
    #     return state, state, txt + ' ' + image_filename + ' '
    

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
                btn = gr.UploadButton("Upload", file_types=["audio"])
        with gr.Column():
            outaudio = gr.Audio()
        txt.submit(bot.run_text, [txt, state], [chatbot, state, outaudio])
        txt.submit(lambda: "", None, txt)
        #btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        btn.upload(bot.run_audio, [btn, state, txt], [chatbot, state, txt, outaudio])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        clear.click(lambda: [], None, outaudio)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
