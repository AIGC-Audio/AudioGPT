import math
import torch


class ExponentialDecayScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, total_iters, final_lrs,
        warmup_iters=3000, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.final_lrs = final_lrs
        if not isinstance(self.final_lrs, list) and not isinstance(
            self.final_lrs, tuple):
            self.final_lrs = [self.final_lrs] * len(optimizer.param_groups)
        self.warmup_iters = warmup_iters
        self.bases = [0.0,] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)
        for i, (base_lr, final_lr) in enumerate(zip(self.base_lrs, self.final_lrs)):
            base = (final_lr / base_lr) ** (1 / (
                self.total_iters - self.warmup_iters))
            self.bases[i] = base

    def _get_closed_form_lr(self):
        warmup_coeff = 1.0
        current_iter = self._step_count
        if current_iter < self.warmup_iters:
            warmup_coeff = current_iter / self.warmup_iters
        current_lrs = []
        # if not self.linear_warmup:
            # for base_lr, final_lr, base in zip(self.base_lrs, self.final_lrs, self.bases):
                # # current_lr = warmup_coeff * base_lr * math.exp(((current_iter - self.warmup_iters) / self.total_iters) * math.log(final_lr / base_lr))
                # current_lr = warmup_coeff * base_lr * (base ** (current_iter - self.warmup_iters))
                # current_lrs.append(current_lr)
        # else:
        for base_lr, final_lr, base in zip(self.base_lrs, self.final_lrs,
                self.bases):
            if current_iter <= self.warmup_iters:
                current_lr = warmup_coeff * base_lr
            else:
                # current_lr = warmup_coeff * base_lr * math.exp(((current_iter - self.warmup_iters) / self.total_iters) * math.log(final_lr / base_lr))
                current_lr = base_lr * (base ** (current_iter - self.warmup_iters))
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, model_size=512, factor=1, warmup_iters=3000,
            last_epoch=-1, verbose=False):
        self.model_size = model_size
        self.warmup_iters = warmup_iters
        # self.factors = [group["lr"] / (self.model_size ** (-0.5) * self.warmup_iters ** (-0.5)) for group in optimizer.param_groups]
        self.factor = factor
        super().__init__(optimizer, last_epoch, verbose)

    def _get_closed_form_lr(self):
        current_iter = self._step_count
        current_lrs = []
        for _ in self.base_lrs:
            current_lr = self.factor * \
                (self.model_size ** (-0.5) * min(current_iter ** (-0.5),
                current_iter * self.warmup_iters ** (-1.5)))
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, total_iters, warmup_iters,
            num_cycles=0.5, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch, verbose)

    def lr_lambda(self, iteration):
        if iteration < self.warmup_iters:
            return float(iteration) / float(max(1, self.warmup_iters))
        progress = float(iteration - self.warmup_iters) / float(max(1,
            self.total_iters - self.warmup_iters))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(
            self.num_cycles) * 2.0 * progress)))

    def _get_closed_form_lr(self):
        current_iter = self._step_count
        current_lrs = []
        for base_lr in self.base_lrs:
            current_lr = base_lr * self.lr_lambda(current_iter)
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()


if __name__ == "__main__":
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)
    epochs = 25
    iters = 600
    scheduler = CosineWithWarmup(optimizer, 600 * 25, 600 * 5,)
    # scheduler = ExponentialDecayScheduler(optimizer, 600 * 25, 5e-7, 600 * 5)
    criterion = torch.nn.MSELoss()
    lrs = []
    for epoch in range(1, epochs + 1):
        for iteration in range(1, iters + 1):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(f"lr: {scheduler.get_last_lr()}")
            # lrs.append(scheduler.get_last_lr())
            lrs.append(optimizer.param_groups[0]["lr"])
    import matplotlib.pyplot as plt
    plt.plot(list(range(1, len(lrs) + 1)), lrs, '-o', markersize=1)
    # plt.legend(loc="best")
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    plt.savefig("lr_curve.png", dpi=100)
