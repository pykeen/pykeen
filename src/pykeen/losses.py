# -*- coding: utf-8 -*-

"""Loss functions integrated in PyKEEN."""

from typing import Any, ClassVar, Mapping, Optional, Set, Type, Union

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
    'MarginRankingLoss',
    'MSELoss',
    'BCEWithLogitsLoss',
    'get_loss_cls',
]

_REDUCTION_METHODS = dict(
    mean=torch.mean,
    sum=torch.sum,
)


class Loss(nn.Module):
    """A loss function."""

    synonyms: ClassVar[Optional[Set[str]]] = None

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}


class PointwiseLoss(Loss):
    """Pointwise loss functions compute an independent loss term for each triple-label pair."""


class PairwiseLoss(Loss):
    """Pairwise loss functions compare the scores of a positive triple and a negative triple."""


class SetwiseLoss(Loss):
    """Setwise loss functions compare the scores of several triples."""


class BCEWithLogitsLoss(PointwiseLoss, nn.BCEWithLogitsLoss):
    r"""A wrapper around the numeric stable version of the PyTorch binary cross entropy loss.

    For label function :math:`l:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \{0,1\}` and interaction
    function :math:`f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}`,
    the binary cross entropy loss is defined as:

    .. math::

        L(h, r, t) = -(l(h,r,t) \cdot \log(\sigma(f(h,r,t))) + (1 - l(h,r,t)) \cdot \log(1 - \sigma(f(h,r,t))))

    where represents the logistic sigmoid function

    .. math::

        \sigma(x) = \frac{1}{1 + \exp(-x)}

    Thus, the problem is framed as a binary classification problem of triples, where the interaction functions' outputs
    are regarded as logits.

    .. warning::

        This loss is not well-suited for translational distance models because these models produce
        a negative distance as score and cannot produce positive model outputs.
    """


class MSELoss(PointwiseLoss, nn.MSELoss):
    """A wrapper around the PyTorch mean square error loss."""

    synonyms = {'Mean Square Error Loss', 'Mean Squared Error Loss'}


class MarginRankingLoss(PairwiseLoss, nn.MarginRankingLoss):
    """A wrapper around the PyTorch margin ranking loss."""

    synonyms = {"Pairwise Hinge Loss"}

    hpo_default = dict(
        margin=dict(type=int, low=0, high=3, q=1),
    )


class SoftplusLoss(PointwiseLoss):
    """A loss function for the softplus."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        self._reduction_method = _REDUCTION_METHODS[reduction]

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Calculate the loss for the given scores and labels."""
        assert 0. <= labels.min() and labels.max() <= 1.
        # scale labels from [0, 1] to [-1, 1]
        labels = 2 * labels - 1
        loss = self.softplus((-1) * labels * logits)
        loss = self._reduction_method(loss)
        return loss


class BCEAfterSigmoidLoss(PointwiseLoss):
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


class CrossEntropyLoss(SetwiseLoss):
    """Evaluate cross entropy after softmax output."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self._reduction_method = _REDUCTION_METHODS[reduction]

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        # cross entropy expects a proper probability distribution -> normalize labels
        p_true = functional.normalize(labels, p=1, dim=-1)
        # Use numerically stable variant to compute log(softmax)
        log_p_pred = logits.log_softmax(dim=-1)
        # compute cross entropy: ce(b) = sum_i p_true(b, i) * log p_pred(b, i)
        sample_wise_cross_entropy = -(p_true * log_p_pred).sum(dim=-1)
        return self._reduction_method(sample_wise_cross_entropy)


class NSSALoss(SetwiseLoss):
    """An implementation of the self-adversarial negative sampling loss function proposed by [sun2019]_."""

    synonyms = {'Self-Adversarial Negative Sampling Loss', 'Negative Sampling Self-Adversarial Loss'}

    hpo_default = dict(
        margin=dict(type=int, low=3, high=30, q=3),
        adversarial_temperature=dict(type=float, low=0.5, high=1.0),
    )

    def __init__(self, margin: float = 9.0, adversarial_temperature: float = 1.0, reduction: str = 'mean') -> None:
        """Initialize the NSSA loss.

        :param margin: The loss's margin (also written as gamma in the reference paper)
        :param adversarial_temperature: The negative sampling temperature (also written as alpha in the reference paper)

        .. note:: The default hyperparameters are based the experiments for FB15K-237 in [sun2019]_.
        """
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
    BCEWithLogitsLoss,
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


#: A mapping of losses' names to their implementations
losses: Mapping[str, Type[Loss]] = {
    normalize_string(cls.__name__, suffix=_LOSS_SUFFIX): cls
    for cls in _LOSSES
}
losses_synonyms: Mapping[str, Type[Loss]] = {
    normalize_string(synonym, suffix=_LOSS_SUFFIX): cls
    for cls in _LOSSES
    if cls.synonyms is not None
    for synonym in cls.synonyms
}


def get_loss_cls(query: Union[None, str, Type[Loss]]) -> Type[Loss]:
    """Get the loss class."""
    return get_cls(
        query,
        base=Loss,
        lookup_dict=losses,
        lookup_dict_synonyms=losses_synonyms,
        default=MarginRankingLoss,
        suffix=_LOSS_SUFFIX,
    )
