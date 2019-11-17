# -*- coding: utf-8 -*-

"""Loss functions implemented in POEM and additionally imported from :class:`torch`.

===============================  ================================================================
Name                             Reference
===============================  ================================================================
bce                              :class:`torch.nn.BCELoss`
bceaftersigmoid                  :class:`poem.loss_functions.BCEAfterSigmoidLoss`
marginranking                    :class:`torch.nn.MarginRankingLoss`
mse                              :class:`torch.nn.MSELoss`
negativesamplingselfadversarial  :class:`poem.loss_functions.NegativeSamplingSelfAdversarialLoss`
softplus                         :class:`poem.loss_functions.SoftplusLoss`
===============================  ================================================================

.. note:: This table can be re-generated with ``poem ls losses -f rst``
"""

from typing import Any, Mapping, Set, Type, Union

import torch
from torch import nn
from torch.nn import BCELoss, MSELoss, MarginRankingLoss, functional

from .utils import get_cls, normalize_string

__all__ = [
    'Loss',
    'BCEAfterSigmoidLoss',
    'SoftplusLoss',
    'NegativeSamplingSelfAdversarialLoss',
    'losses',
    'losses_hpo_defaults',
    'get_loss_cls',
]

Loss = nn.modules.loss._Loss

_REDUCTION_METHODS = dict(
    mean=torch.mean,
    sum=torch.sum,
)


class SoftplusLoss(nn.Module):
    """A loss function for the softplus."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        self._reduction_method = _REDUCTION_METHODS[reduction]

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Calculate the loss for the given scores and labels."""
        loss = self.softplus((-1) * labels * scores)
        loss = self._reduction_method(loss)
        return loss


class BCEAfterSigmoidLoss(nn.Module):
    """A loss function which uses the numerically unstable version of explicit Sigmoid + BCE."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        post_sigmoid = torch.sigmoid(logits)
        return functional.binary_cross_entropy(post_sigmoid, labels, **kwargs)


class NegativeSamplingSelfAdversarialLoss(nn.Module):
    """An implementation of the self-adversarial negative sampling loss function proposed by [sun2019]_."""

    def __init__(self, margin: float, adversarial_temperature: float, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.adversarial_temperature = adversarial_temperature
        self.margin = margin
        self._reduction_method = _REDUCTION_METHODS[reduction]

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Calculate the loss for the given scores.

        .. seealso:: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py
        """
        neg_score_weights = functional.softmax(neg_scores * self.adversarial_temperature, dim=-1).detach()
        neg_distances = -neg_scores
        weighted_neg_scores = neg_score_weights * functional.logsigmoid(neg_distances - self.margin)
        neg_loss = self._reduction_method(weighted_neg_scores)
        pos_distances = -pos_scores
        pos_loss = self._reduction_method(functional.logsigmoid(self.margin - pos_distances))
        loss = -pos_loss - neg_loss

        if self._reduction_method is torch.mean:
            loss = loss / 2.

        return loss


_LOSS_SUFFIX = 'Loss'
_LOSSES: Set[Type[Loss]] = {
    MarginRankingLoss,
    BCELoss,
    SoftplusLoss,
    BCEAfterSigmoidLoss,
    MSELoss,
    NegativeSamplingSelfAdversarialLoss,
}
# To add *all* losses implemented in Torch, uncomment:
# _LOSSES.update({
#     criterion
#     for criterion in Loss.__subclasses__() + WeightedLoss.__subclasses__()
#     if not criterion.__name__.startswith('_')
# })


losses: Mapping[str, Type[Loss]] = {
    normalize_string(cls.__name__, suffix=_LOSS_SUFFIX): cls
    for cls in _LOSSES
}

losses_hpo_defaults: Mapping[Type[Loss], Mapping[str, Any]] = {
    MarginRankingLoss: dict(
        margin=dict(type=int, low=0, high=3, q=1),
    ),
}
# Add empty dictionaries as defaults for all remaining criteria
for criterion in _LOSSES:
    if criterion not in losses_hpo_defaults:
        losses_hpo_defaults[criterion] = {}


def get_loss_cls(query: Union[None, str, Type[Loss]]) -> Type[Loss]:
    """Get the loss class."""
    return get_cls(
        query,
        base=nn.Module,
        lookup_dict=losses,
        default=MarginRankingLoss,
        suffix=_LOSS_SUFFIX,
    )
