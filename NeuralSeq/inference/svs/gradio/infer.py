import importlib
import re

import gradio as gr
import yaml
from gradio.inputs import Textbox

from inference.svs.base_svs_infer import BaseSVSInfer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import numpy as np


class GradioInfer:
    def __init__(self, exp_name, inference_cls, title, description, article, example_inputs):
        self.exp_name = exp_name
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs
        pkg = ".".join(inference_cls.split(".")[:-1])
        cls_name = inference_cls.split(".")[-1]
        self.inference_cls = getattr(importlib.import_module(pkg), cls_name)

    def greet(self, text, notes, notes_duration):
        PUNCS = '。？；：'
        sents = re.split(rf'([{PUNCS}])', text.replace('\n', ','))
        sents_notes = re.split(rf'([{PUNCS}])', notes.replace('\n', ','))
        sents_notes_dur = re.split(rf'([{PUNCS}])', notes_duration.replace('\n', ','))

        if sents[-1] not in list(PUNCS):
            sents = sents + ['']
            sents_notes = sents_notes + ['']
            sents_notes_dur = sents_notes_dur + ['']

        audio_outs = []
        s, n, n_dur = "", "", ""
        for i in range(0, len(sents), 2):
            if len(sents[i]) > 0:
                s += sents[i] + sents[i + 1]
                n += sents_notes[i] + sents_notes[i+1]
                n_dur += sents_notes_dur[i] + sents_notes_dur[i+1]
            if len(s) >= 400 or (i >= len(sents) - 2 and len(s) > 0):
                audio_out = self.infer_ins.infer_once({
                    'text': s,
                    'notes': n,
                    'notes_duration': n_dur,
                })
                audio_out = audio_out * 32767
                audio_out = audio_out.astype(np.int16)
                audio_outs.append(audio_out)
                audio_outs.append(np.zeros(int(hp['audio_sample_rate'] * 0.3)).astype(np.int16))
                s = ""
                n = ""
        audio_outs = np.concatenate(audio_outs)
        return hp['audio_sample_rate'], audio_outs

    def run(self):
        set_hparams(exp_name=self.exp_name, print_hparams=False)
        infer_cls = self.inference_cls
        self.infer_ins: BaseSVSInfer = infer_cls(hp)
        example_inputs = self.example_inputs
        for i in range(len(example_inputs)):
            text, notes, notes_dur = example_inputs[i].split('<sep>')
            example_inputs[i] = [text, notes, notes_dur]

        iface = gr.Interface(fn=self.greet,
                             inputs=[
                                 Textbox(lines=2, placeholder=None, default=example_inputs[0][0], label="input text"),
                                 Textbox(lines=2, placeholder=None, default=example_inputs[0][1], label="input note"),
                                 Textbox(lines=2, placeholder=None, default=example_inputs[0][2], label="input duration")]
                             ,
                             outputs="audio",
                             allow_flagging="never",
                             title=self.title,
                             description=self.description,
                             article=self.article,
                             examples=example_inputs,
                             enable_queue=True)
        iface.launch(share=True,)# cache_examples=True)


if __name__ == '__main__':
    gradio_config = yaml.safe_load(open('inference/svs/gradio/gradio_settings.yaml'))
    g = GradioInfer(**gradio_config)
    g.run()


# python inference/svs/gradio/infer.py --config usr/configs/midi/cascade/opencs/ds60_rel.yaml --exp_name 0303_opencpop_ds58_midi
# python inference/svs/ds_cascade.py --config usr/configs/midi/cascade/opencs/ds60_rel.yaml --exp_name 0303_opencpop_ds58_midi
# CUDA_VISIBLE_DEVICES=3 python inference/svs/gradio/infer.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name 0228_opencpop_ds100_rel