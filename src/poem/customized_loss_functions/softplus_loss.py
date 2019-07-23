# -*- coding: utf-8 -*-

import torch
from torch._jit_internal import weak_script_method
from torch.nn.modules.loss import _Loss

__all__ = [
    'SoftplusLoss',
]


class SoftplusLoss(_Loss):

    def __init__(self, reduction='mean'):
        super(SoftplusLoss, self).__init__(reduction=reduction)
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        if self.reduction == 'mean':
            self._reduction_method = torch.mean
        else:
            self._reduction_method = torch.sum

    @weak_script_method
    def forward(self, scores, labels):
        loss = self.softplus((-1) * labels * scores)
        loss = self._reduction_method(loss)
        return loss
