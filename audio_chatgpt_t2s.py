import sys
import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"
import uuid
import torch
import numpy as np
import re
from scipy.io import wavfile
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_sing/DiffSinger'))
print(sys.path)
import torch

from text_to_sing.DiffSinger.inference.svs.ds_e2e import DiffSingerE2EInfer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
from utils.text_encoder import TokenTextEncoder
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import gradio as gr
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

temp_audio_filename = "audio/text2sing.wav"
os.makedirs('audio', exist_ok=True)
def cut_dialogue_history(history_memory, keep_last_n_words=500):
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

class test:
    def __init__(self, device= None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def inference(self, inputs):
        global temp_audio_filename
        key = ['text', 'notes', 'notes_duration']
        val = inputs.split(",")
        inp = {k:v for k,v in zip(key,val)}
        wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        temp_audio_filename = audio_filename
        wavfile.write(temp_audio_filename, hp['audio_sample_rate'], wav.astype(np.int16))
        print(f"Processed T2S.run, text: {val[0]}, notes: {val[1]}, notes duration: {val[2]}, audio_filename: {audio_filename}")
        return temp_audio_filename
class T2S:
    def __init__(self, device= None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing DiffSinger to %s" % device)
        self.device = device
        exp_name = 'text_to_sing/DiffSinger/checkpoints/0831_opencpop_ds1000'
        config= 'text_to_sing/DiffSinger/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml'
        set_hparams(config= config,exp_name=exp_name, print_hparams=False)
        self.pipe = DiffSingerE2EInfer(hp)

    def inference(self, inputs):
        global temp_audio_filename
        key = ['text', 'notes', 'notes_duration']
        val = inputs.split(",")
        inp = {k:v for k,v in zip(key,val)}
        print(f"\nDiffsinger input: {inp}")
        wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        temp_audio_filename = audio_filename
        wavfile.write(temp_audio_filename, hp['audio_sample_rate'], wav.astype(np.int16))
        print(f"Processed T2S.run, text: {val[0]}, notes: {val[1]}, notes duration: {val[2]}, audio_filename: {audio_filename}")
        return temp_audio_filename


class ConversationBot:
    def __init__(self):
        print("Initializing AudioChatGPT")
        self.llm = OpenAI(temperature=0)

        #self.t2i = T2I(device="cuda:0")
        #self.t2a = T2A(device="cuda:0")
        self.t2s = T2S(device="cuda:0")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            #Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
            #     description="useful for when you want to generate an image from a user input text and it saved it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
            #                 "The input to this tool should be a string, representing the text used to generate image. "),
            Tool(name="Generate singing voice From User Input Text", func=self.t2s.inference,
                 description="useful for when you want to generate a piece of singing voice from its description."
                             "The input to this tool should be a comma seperated string of three, representing the text sequence and its corresponding note and duration sequence."
                             "Text sequence consists of Chinese characters (except for SP and AP). "
                             "Each component of the note and duration sequence sequences should be separated by | mark."
                             "It is necessary to ensure that note and duration sequence is of the same length. ")]
            #Tool(name="Generate Audio From User Input Text", func=self.t2a.inference,
            #     description="useful for when you want to generate an audio from a user input text and it saved it to a file."
            #                 "The input to this tool should be a string, representing the text used to generate audio.")]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': AUDIO_CHATGPT_PREFIX, 'format_instructions': AUDIO_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': AUDIO_CHATGPT_SUFFIX}, )

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
        # return outaudio

    def run_image(self, image, state, txt):
        print("===============Running run_image =============")
        print("Inputs:", image, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.i2t.inference(image_filename)
        Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(
            image_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + ' ' + image_filename + ' '




if __name__ == '__main__':
    inp = {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
        'note_seq': 'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 F#4/Gb4 F#4/Gb4 F#4/Gb4 C#4/Db4 C#4/Db4 C#4/Db4 rest C#4/Db4 C#4/Db4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 F4 F4 C#4/Db4 C#4/Db4',
        'note_dur_seq': '0.407140 0.407140 0.376190 0.376190 0.242180 0.242180 0.509550 0.509550 0.183420 0.315400 0.315400 0.235020 0.361660 0.361660 0.223070 0.377270 0.377270 0.340550 0.340550 0.299620 0.299620 0.344510 0.344510 0.283770 0.283770 0.323390 0.323390 0.360340 0.360340',
        'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
        'input_type': 'phoneme'
    }  # input like Opencpop dataset.

    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        with gr.Row():
            gr.Markdown("## Audio ChatGPT")
        chatbot = gr.Chatbot(elem_id="chatbot", label="Audio ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear️")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])
        with gr.Column():
            outaudio = gr.Audio()
        txt.submit(bot.run_text, [txt, state], [chatbot, state, outaudio])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
