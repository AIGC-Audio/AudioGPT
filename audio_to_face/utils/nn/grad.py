import torch

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
    if isinstance(accu_grad, float):
        return accu_grad
    return accu_grad.item()

class GradBuffer:
    def __init__(self):
        self.buffer = {}
    
    def add(self, model):
        for item in model.named_parameters():
            name, param = item
            if param.grad is None:
                continue
            self.buffer[name] = self.buffer.get(name, 0) + param.grad.data
    
    def apply(self, model):
        for item in model.named_parameters():
            name, param = item
            if param.grad is None:
                continue
            if name in self.buffer.keys():
                param.grad.data += self.buffer[name]
        self.buffer = {}