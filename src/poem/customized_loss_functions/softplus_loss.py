import torch
from torch._jit_internal import weak_script_method
from torch.nn.modules.loss import _Loss


class SoftplusLoss(_Loss):

    def __init__(self, reduction='mean'):
        super(SoftplusLoss, self).__init__(reduction=reduction)

    @weak_script_method
    def forward(self, scores, labels):
        """."""
        labels = labels * (-1)
        scores = labels * scores
        scores = torch.exp(scores)
        scores += 1.

        scores = torch.log(scores)
        loss = torch.mean(scores) if self.reduction == 'mean' else torch.sum(scores)

        return loss
