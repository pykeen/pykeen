# -*- coding: utf-8 -*-

"""A loss function for the softplus."""

import logging

import torch
from torch.nn.modules.loss import _Loss


try:
    from torch._jit_internal import weak_script_method
except ImportError:
    logging.warn('torch._jit_internal is not available')

    def weak_script_method(f):
        """Return the original function because weak_script_method is not available."""
        return f

__all__ = [
    'SoftplusLoss',
]


class SoftplusLoss(_Loss):
    """A loss function for the softplus."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        if self.reduction == 'mean':
            self._reduction_method = torch.mean
        else:
            self._reduction_method = torch.sum

    @weak_script_method
    def forward(self, scores, labels):
        """Calculate the loss for the given scores and labels."""
        loss = self.softplus((-1) * labels * scores)
        loss = self._reduction_method(loss)
        return loss
