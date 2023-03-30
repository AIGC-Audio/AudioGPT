import time
import torch


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Timer:
    timer_map = {}

    def __init__(self, name, enable=False):
        if name not in Timer.timer_map:
            Timer.timer_map[name] = 0
        self.name = name
        self.enable = enable

    def __enter__(self):
        if self.enable:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            Timer.timer_map[self.name] += time.time() - self.t
            if self.enable:
                print(f'[Timer] {self.name}: {Timer.timer_map[self.name]}')
