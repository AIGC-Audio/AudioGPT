# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
from typing import Callable, Dict, Union
import yaml
import torch
from torch.optim.swa_utils import AveragedModel as torch_average_model
import numpy as np
import pandas as pd
from pprint import pformat


def load_dict_from_csv(csv, cols):
    df = pd.read_csv(csv, sep="\t")
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


def init_logger(filename, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + filename)
    logger.setLevel(getattr(logging, level))
    # Log results to std
    # stdhandler = logging.StreamHandler(sys.stdout)
    # stdhandler.setFormatter(formatter)
    # Dump log to file
    filehandler = logging.FileHandler(filename)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(stdhandler)
    return logger


def init_obj(module, config, **kwargs):# 'captioning.models.encoder'
    obj_args = config["args"].copy()
    obj_args.update(kwargs)
    return getattr(module, config["type"])(**obj_args)


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_config(config_file):
    with open(config_file, "r") as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    if "inherit_from" in config:
        base_config_file = config["inherit_from"]
        base_config_file = os.path.join(
            os.path.dirname(config_file), base_config_file
        )
        assert not os.path.samefile(config_file, base_config_file), \
            "inherit from itself"
        base_config = load_config(base_config_file)
        del config["inherit_from"]
        merge_a_into_b(config, base_config)
        return base_config
    return config


def parse_config_or_kwargs(config_file, **kwargs):
    yaml_config = load_config(config_file)
    # passed kwargs will override yaml config
    args = dict(yaml_config, **kwargs)
    return args


def store_yaml(config, config_file):
    with open(config_file, "w") as con_writer:
        yaml.dump(config, con_writer, indent=4, default_flow_style=False)


class MetricImprover:

    def __init__(self, mode):
        assert mode in ("min", "max")
        self.mode = mode
        # min: lower -> better; max: higher -> better
        self.best_value = np.inf if mode == "min" else -np.inf

    def compare(self, x, best_x):
        return x < best_x if self.mode == "min" else x > best_x

    def __call__(self, x):
        if self.compare(x, self.best_value):
            self.best_value = x
            return True
        return False

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def fix_batchnorm(model: torch.nn.Module):
    def inner(module):
        class_name = module.__class__.__name__
        if class_name.find("BatchNorm") != -1:
            module.eval()
    model.apply(inner)


def load_pretrained_model(model: torch.nn.Module,
                          pretrained: Union[str, Dict],
                          output_fn: Callable = sys.stdout.write):
    if not isinstance(pretrained, dict) and not os.path.exists(pretrained):
        output_fn(f"pretrained {pretrained} not exist!")
        return
    
    if hasattr(model, "load_pretrained"):
        model.load_pretrained(pretrained)
        return

    if isinstance(pretrained, dict):
        state_dict = pretrained
    else:
        state_dict = torch.load(pretrained, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (
            model_dict[k].shape == v.shape)
    }
    output_fn(f"Loading pretrained keys {pretrained_dict.keys()}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)


class AveragedModel(torch_average_model):

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))

        for b_swa, b_model in zip(list(self.buffers())[1:], model.buffers()):
            device = b_swa.device
            b_model_ = b_model.detach().to(device)
            if self.n_averaged == 0:
                b_swa.detach().copy_(b_model_)
            else:
                b_swa.detach().copy_(self.avg_fn(b_swa.detach(), b_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1
