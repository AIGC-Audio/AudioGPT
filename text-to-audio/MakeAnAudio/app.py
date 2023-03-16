import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
from vocoder.hifigan.modules import VocoderHifigan
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from wav_evaluation.models.CLAPWrapper import CLAPWrapper

SAMPLE_RATE = 16000

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device,device,model.cond_stage_model.device)
    sampler = DDIMSampler(model)

    return sampler

sampler = initialize_model('configs/text_to_audio/txt2audio_args.yaml', 'useful_ckpts/ta40multi_epoch=000085.ckpt')
vocoder = VocoderHifigan('vocoder/logs/hifi_0127',device=device)
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

def txt2audio(sampler,vocoder,prompt, seed, scale, ddim_steps, n_samples=1, W=624, H=80):
    prng = np.random.RandomState(seed)
    start_code = prng.randn(n_samples, sampler.model.first_stage_model.embed_dim, H // 8, W // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)
    
    uc = None
    if scale != 1.0:
        uc = sampler.model.get_learned_conditioning(n_samples * [""])
    c = sampler.model.get_learned_conditioning(n_samples * [prompt])
    shape = [sampler.model.first_stage_model.embed_dim, H//8, W//8]  # (z_dim, 80//2^x, 848//2^x)
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        x_T=start_code)

    x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0) # [0, 1]

    wav_list = []
    for idx,spec in enumerate(x_samples_ddim):
        wav = vocoder.vocode(spec)
        wav_list.append((SAMPLE_RATE,wav))
    best_wav = select_best_audio(prompt,wav_list)
    return best_wav


def predict(prompt, ddim_steps, num_samples, scale, seed):
    melbins,mel_len = 80,624
    with torch.no_grad():
        result = txt2audio(
            sampler=sampler,
            vocoder=vocoder,
            prompt=prompt,
            seed=seed,
            scale=scale,
            ddim_steps=ddim_steps,
            n_samples=num_samples,
            H=melbins, W=mel_len
        )

    return result


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Make-An-Audio: Text-to-Audio Generation")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt: Input your text here:")
            run_button = gr.Button(label="Run")

            
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Candidates", minimum=1, maximum=10, value=3, step=1)
                # num_samples = 1
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=150, value=100, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=4.0, value=1.5, step=0.1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    value=44,
                )

        with gr.Column():
            # audio_list = []
            # for i in range(int(num_samples)):
            #     audio_list.append(gr.outputs.Audio())
            outaudio = gr.Audio()


    run_button.click(fn=predict, inputs=[
                    prompt,ddim_steps, num_samples, scale, seed], outputs=[outaudio])# inputs的参数只能传gr.xxx
    with gr.Row():
        with gr.Column():
            gr.Examples(
                        examples = [['a dog barking and a bird chirping',100,3,1.5,55],['fireworks pop and explode',100,3,1.5,55],
                                        ['piano and violin plays',100,3,1.5,55],['wind thunder and rain falling',100,3,1.5,55],['music made by drum kit',100,3,1.5,55]],
                        inputs = [prompt,ddim_steps, num_samples, scale, seed],
                        outputs = [outaudio]
                        )
        with gr.Column():
            pass

demo.launch()
