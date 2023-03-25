import numpy as np


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
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-6)
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
        if num_updates < self.warmup_updates:
            lr = self._warmup_lr(self.lr, self.warmup_updates, num_updates)
        else:
            e = num_updates - self.warmup_updates
            es = self.total_updates - self.warmup_updates
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.lr
        self.assign_learning_rate(self.optimizer, lr)
        return lr


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def plot_scheduler(scheduler, label=None):
        y = np.array([scheduler.step(x) for x in range(0,160000, 10)])
        x = np.arange(0,160000, 10)
        plt.plot(x, y, label=label)

    dummy_model = torch.nn.Linear(10,10)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    rsqrt = CosineSchedule(dummy_optimizer, lr=0.0005, warmup_updates=10000, total_updates=160000)
    plot_scheduler(rsqrt, "8000")
    plt.savefig("0.png")
