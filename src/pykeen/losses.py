# -*- coding: utf-8 -*-

"""Loss functions implemented in PyKEEN and additionally imported from :class:`torch`.

===============  ==========================================
Name             Reference
===============  ==========================================
bce              :class:`torch.nn.BCELoss`
bceaftersigmoid  :class:`pykeen.losses.BCEAfterSigmoidLoss`
crossentropy     :class:`pykeen.losses.CrossEntropyLoss`
marginranking    :class:`torch.nn.MarginRankingLoss`
mse              :class:`torch.nn.MSELoss`
nssa             :class:`pykeen.losses.NSSALoss`
softplus         :class:`pykeen.losses.SoftplusLoss`
===============  ==========================================

.. note:: This table can be re-generated with ``pykeen ls losses -f rst``
"""

from typing import Any, Callable, Mapping, Set, Type, Union

import torch
from torch import nn
from torch.nn import functional

from .utils import get_cls, normalize_string

__all__ = [
    'Loss',
    'BCEAfterSigmoidLoss',
    'SoftplusLoss',
    'NSSALoss',
    'CrossEntropyLoss',
    'losses',
    'losses_hpo_defaults',
    'get_loss_cls',
]

# Loss = nn.modules.loss._Loss

_REDUCTION_METHODS = dict(
    mean=torch.mean,
    sum=torch.sum,
)


class Loss(nn.Module):
    """A base class for losses for link prediction."""

    def __init__(
        self,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.reduction = reduction

    @property
    def reduction_operation(self) -> Callable[[torch.FloatTensor], torch.FloatTensor]:
        """Return the reduction operation."""
        return _REDUCTION_METHODS[self.reduction]


class PointwiseLoss(Loss):
    """Base class for point-wise losses.

    These losses receive the score of a triple together with its label."""

    def forward(
        self,
        score: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the loss function.

        :param score: (batch_size,)
            The individual triple scores.
        :param labels:  (batch_size,)
            The corresponding labels in [0, 1].

        :return:
            A scalar loss value.
        """
        raise NotImplementedError


class BCELoss(PointwiseLoss):
    """The binary cross entropy loss directly calculated from logits."""

    def forward(
        self,
        score: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert labels.min() >= 0 and labels.max() <= 1
        return functional.binary_cross_entropy_with_logits(score, labels, reduction=self.reduction)


class BCEAfterSigmoidLoss(PointwiseLoss):
    """A loss function which uses the numerically unstable version of explicit Sigmoid + BCE."""

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return functional.binary_cross_entropy(torch.sigmoid(logits), labels, reduction=self.reduction)


class MSELoss(PointwiseLoss):
    """The MSE loss."""

    def forward(
        self,
        score: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert labels.min() >= 0 and labels.max() <= 1
        return functional.mse_loss(score, labels, reduction=self.reduction)


class SoftplusLoss(PointwiseLoss):
    """A loss function using softplus."""

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert labels.min() >= 0 and labels.max() <= 1
        # scale labels from [0, 1] to [-1, 1]
        labels = 2 * labels - 1
        loss = functional.softplus((-1) * labels * logits)
        return self.reduction_operation(loss)


class PairwiseLoss(Loss):
    """Base class for pair-wise losses.

    These losses consider a pair of a positive and negative score.
    """

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the loss function.

        :param pos_scores: shape: (batch_size,)
            Scores for positive triples.
        :param neg_scores: shape: (batch_size, num_neg_per_pos)
            Score for negative triples. There may be more than one negative for each positive.

        :return:
            A scalar loss value.
        """
        raise NotImplementedError


class MarginRankingLoss(PairwiseLoss):
    """The margin ranking loss."""

    def __init__(
        self,
        margin: float = 1.0,
        margin_activation: Callable[[torch.FloatTensor], torch.FloatTensor] = functional.relu,
        reduction: str = 'mean',
    ):
        super().__init__(reduction=reduction)
        self.margin = margin
        self.margin_activation = margin_activation

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.reduction_operation(self.margin_activation(neg_scores - pos_scores.unsqueeze(dim=-1) + self.margin))


class SetwiseLoss(Loss):
    """Base class for set-wise losses.

    These losses consider the whole set of triple scores."""

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the loss function.

        :param scores: shape: (batch_size, num_triples_per_batch)
            The triple scores.
        :param labels: shape: (batch_size, num_triples_per_batch)
            The labels for each triple, in [0, 1].
        """
        raise NotImplementedError


class CrossEntropyLoss(SetwiseLoss):
    """Evaluate cross entropy after softmax output."""

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        # cross entropy expects a proper probability distribution -> normalize labels
        p_true = functional.normalize(labels, p=1, dim=-1)

        # Use numerically stable variant to compute log(softmax)
        log_p_pred = scores.log_softmax(dim=-1)

        # compute cross entropy: ce(b) = sum_i p_true(b, i) * log p_pred(b, i)
        sample_wise_cross_entropy = -(p_true * log_p_pred).sum(dim=-1)
        return self.reduction_operation(sample_wise_cross_entropy)


class NSSALoss(PairwiseLoss):
    """An implementation of the self-adversarial negative sampling loss function proposed by [sun2019]_."""

    # TODO: Actually the loss is pointwise. It is only the weighting, which is setwise on the negative triples.

    def __init__(self, margin: float, adversarial_temperature: float, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.adversarial_temperature = adversarial_temperature
        self.margin = margin

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
        neg_loss = self.reduction_operation(weighted_neg_scores)
        pos_distances = -pos_scores
        pos_loss = self.reduction_operation(functional.logsigmoid(self.margin - pos_distances))
        loss = -pos_loss - neg_loss

        if self.reduction == 'mean':
            loss = loss / 2.

        return loss


_LOSS_SUFFIX = 'Loss'
_LOSSES: Set[Type[Loss]] = {
    MarginRankingLoss,
    BCELoss,
    SoftplusLoss,
    BCEAfterSigmoidLoss,
    CrossEntropyLoss,
    MSELoss,
    NSSALoss,
}
# To add *all* losses implemented in Torch, uncomment:
# _LOSSES.update({
#     loss
#     for loss in Loss.__subclasses__() + WeightedLoss.__subclasses__()
#     if not loss.__name__.startswith('_')
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
# Add empty dictionaries as defaults for all remaining losses
for cls in _LOSSES:
    if cls not in losses_hpo_defaults:
        losses_hpo_defaults[cls] = {}


def get_loss_cls(query: Union[None, str, Type[Loss]]) -> Type[Loss]:
    """Get the loss class."""
    return get_cls(
        query,
        base=Loss,
        lookup_dict=losses,
        default=MarginRankingLoss,
        suffix=_LOSS_SUFFIX,
    )
