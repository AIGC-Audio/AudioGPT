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

def get_grad_norm(model, l=2):
    num_para = 0
    accu_grad = 0
    if isinstance(model, torch.nn.Module):
        params = model.parameters()
    else:
        params = model
    for p in params:
        if p.grad is None:
            continue
        num_para += p.numel()
        if l == 1:
            accu_grad += p.grad.abs(1).sum()
        elif l == 2:
            accu_grad += p.grad.pow(2).sum()
        else:
            raise ValueError("Now we only implement l1/l2 norm !")
    if l == 2:
        accu_grad = accu_grad ** 0.5
    return accu_grad