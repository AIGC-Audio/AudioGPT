import numpy as np
from audio_to_face.utils.commons.hparams import hparams


class NoneSchedule(object):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.step(0)

    def step(self, num_updates):
        self.lr = self.constant_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        return self.get_lr()


class RSQRTSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates, hidden_size):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.warmup_updates = warmup_updates
        self.hidden_size = hidden_size
        self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr


class WarmupSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = warmup_updates
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        self.lr = max(constant_lr * warmup, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr


class ExponentialSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = warmup_updates
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 250k steps
            self.lr = max(new_lrate, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr


class ExponentialScheduleWithAudattNet(NoneSchedule):
    """
    Default Scheduler in AD-NeRF
    for audatt net, since it starts at 20_0000 steps, we need to enlarge its lr
    in optimizer, we set param_groups[1] to optimize audatt net
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = warmup_updates
        optimizer.param_groups[0]['lr'] = self.lr
        optimizer.param_groups[1]['lr'] = self.lr * 5
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 250k steps
            self.lr = max(new_lrate, 1e-7)

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 5
        return self.lr

class ExponentialScheduleForRADNeRF(NoneSchedule):
    """
    Default Scheduler in RAD-NeRF
    RAD-NeRF has two groups of params with different lr
    for tileGrid embedding, the lr=5e-3
    for other network params, the lr=5e-4
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr # 0.0005
        self.warmup_updates = warmup_updates
        self.finetune_lips = hparams['finetune_lips']
        self.finetune_lips_start_iter = hparams['finetune_lips_start_iter']

        optimizer.param_groups[0]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        optimizer.param_groups[1]['lr'] = self.lr * 10 # for tileGrid, lr starts from 0.005
        optimizer.param_groups[2]['lr'] = self.lr * 5 # for Att Net, lr starts from 0.0025
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            if self.finetune_lips and num_updates > self.finetune_lips_start_iter:
                new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.05x for every 200k steps
            else:
                new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 200k steps

            self.lr = max(new_lrate, 1e-7)

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 10
        self.optimizer.param_groups[2]['lr'] = self.lr * 5
        return self.lr
    

class ExponentialScheduleForRADNeRFTorso(NoneSchedule):
    """
    Default Scheduler in RAD-NeRF
    RAD-NeRF has two groups of params with different lr
    for tileGrid embedding, the lr=5e-3
    for other network params, the lr=5e-4
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr # 0.0005
        self.warmup_updates = warmup_updates

        optimizer.param_groups[0]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        optimizer.param_groups[1]['lr'] = self.lr * 10 # for tileGrid, lr starts from 0.005
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 200k steps
            self.lr = max(new_lrate, 1e-7)
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 10
        return self.lr
    

class CosineSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates, total_updates):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        self.lr = lr
        self.assign_learning_rate(self.optimizer, self.lr)
        self.step(0)

    def assign_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _warmup_lr(self, base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def step(self, num_updates):
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            lr = self._warmup_lr(self.lr, self.warmup_updates, num_updates)
        else:
            e = num_updates - self.warmup_updates
            es = self.total_updates - self.warmup_updates
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.lr
        self.assign_learning_rate(self.optimizer, lr)
        return lr
