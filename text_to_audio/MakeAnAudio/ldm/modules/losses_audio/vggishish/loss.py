import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WeightedCrossEntropy(nn.CrossEntropyLoss):

    def __init__(self, weights, **pytorch_ce_loss_args) -> None:
        super().__init__(reduction='none', **pytorch_ce_loss_args)
        self.weights = weights

    def __call__(self, outputs, targets, to_weight=True):
        loss = super().__call__(outputs, targets)
        if to_weight:
            return (loss * self.weights[targets]).sum() / self.weights[targets].sum()
        else:
            return loss.mean()


if __name__ == '__main__':
    x = torch.randn(10, 5)
    target = torch.randint(0, 5, (10,))
    weights = torch.tensor([1., 2., 3., 4., 5.])

    # criterion_weighted = nn.CrossEntropyLoss(weight=weights)
    # loss_weighted = criterion_weighted(x, target)

    # criterion_weighted_manual = nn.CrossEntropyLoss(reduction='none')
    # loss_weighted_manual = criterion_weighted_manual(x, target)
    # print(loss_weighted, loss_weighted_manual.mean())
    # loss_weighted_manual = (loss_weighted_manual * weights[target]).sum() / weights[target].sum()
    # print(loss_weighted, loss_weighted_manual)
    # print(torch.allclose(loss_weighted, loss_weighted_manual))

    pytorch_weighted = nn.CrossEntropyLoss(weight=weights)
    pytorch_unweighted = nn.CrossEntropyLoss()
    custom = WeightedCrossEntropy(weights)

    assert torch.allclose(pytorch_weighted(x, target), custom(x, target, to_weight=True))
    assert torch.allclose(pytorch_unweighted(x, target), custom(x, target, to_weight=False))
    print(custom(x, target, to_weight=True), custom(x, target, to_weight=False))
