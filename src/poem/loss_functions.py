# -*- coding: utf-8 -*-

"""Custom loss functions."""

from typing import List, Mapping, Type, Union

import torch
from torch import nn
from torch.nn import BCELoss, MarginRankingLoss, functional

from .typing import Loss
from .utils import get_cls, normalize_string

__all__ = [
    'BCEAfterSigmoid',
    'SoftplusLoss',
    'losses',
    'get_loss_cls',
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
        post_sigmoid = torch.sigmoid(logits)
        return functional.binary_cross_entropy(post_sigmoid, labels, **kwargs)


_LOSSES_LIST: List[Type[Loss]] = [
    MarginRankingLoss,
    BCELoss,
    SoftplusLoss,
    BCEAfterSigmoid,
]

losses: Mapping[str, Type[Loss]] = {
    normalize_string(criterion.__name__): criterion
    for criterion in _LOSSES_LIST
}


def get_loss_cls(query: Union[None, str, Type[Loss]]) -> Type[Loss]:
    """Get the loss class."""
    return get_cls(
        query,
        base=nn.Module,
        lookup_dict=losses,
        default=MarginRankingLoss,
    )
