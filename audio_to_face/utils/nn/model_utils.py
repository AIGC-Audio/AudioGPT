import numpy as np
import torch


def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    num_params(model, model_name=model_name)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters

def get_device_of_model(model):
    return model.parameters().__next__().device

def requires_grad(model):
    if isinstance(model, torch.nn.Module):
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.requires_grad = True

def not_requires_grad(model):
    if isinstance(model, torch.nn.Module):
        for p in model.parameters():
            p.requires_grad = False
    else:
        model.requires_grad = False
