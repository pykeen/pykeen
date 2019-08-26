# -*- coding: utf-8 -*-

"""Custom loss functions."""

import torch
from torch import nn
from torch.nn import functional

__all__ = [
    'BCEAfterSigmoid',
    'SoftplusLoss',
]


class SoftplusLoss(nn.Module):
    """A loss function for the softplus."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        if reduction == 'mean':
            self._reduction_method = torch.mean
        else:
            self._reduction_method = torch.sum

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Calculate the loss for the given scores and labels."""
        loss = self.softplus((-1) * labels * scores)
        loss = self._reduction_method(loss)
        return loss


class BCEAfterSigmoid(nn.Module):
    """A loss function which uses the numerically unstable version of explicit Sigmoid + BCE."""

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        post_sigmoid = functional.sigmoid(logits)
        return functional.binary_cross_entropy(post_sigmoid, labels, **kwargs)
