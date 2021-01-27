# -*- coding: utf-8 -*-

"""Loss functions integrated in PyKEEN."""

from typing import Any, Callable, ClassVar, Mapping, Optional, Set, Type, Union

import torch
from torch.nn import functional
from torch.nn.modules.loss import _Loss

from .utils import get_cls, normalize_string

__all__ = [
    # Helpers
    'get_loss_cls',
    # Base Classes
    'Loss',
    'PointwiseLoss',
    'PairwiseLoss',
    'SetwiseLoss',
    # Concrete Classes
    'BCEAfterSigmoidLoss',
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
    'MarginRankingLoss',
    'MSELoss',
    'NSSALoss',
    'SoftplusLoss',
    'has_mr_loss',
    'has_nssa_loss',
]

_REDUCTION_METHODS = dict(
    mean=torch.mean,
    sum=torch.sum,
)


class Loss(_Loss):
    """A loss function."""

    synonyms: ClassVar[Optional[Set[str]]] = None

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self._reduction_method = _REDUCTION_METHODS[reduction]


class PointwiseLoss(Loss):
    """Pointwise loss functions compute an independent loss term for each triple-label pair."""

    @staticmethod
    def validate_labels(labels: torch.FloatTensor) -> bool:
        """Check whether labels are in [0, 1]."""
        return labels.min() >= 0 and labels.max() <= 1


class PairwiseLoss(Loss):
    """Pairwise loss functions compare the scores of a positive triple and a negative triple."""


class SetwiseLoss(Loss):
    """Setwise loss functions compare the scores of several triples."""


class BCEWithLogitsLoss(PointwiseLoss):
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

    synonyms = {'Negative Log Likelihood Loss'}

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return functional.binary_cross_entropy_with_logits(scores, labels, reduction=self.reduction)


class MSELoss(PointwiseLoss):
    """A wrapper around the PyTorch mean square error loss."""

    synonyms = {'Mean Square Error Loss', 'Mean Squared Error Loss'}

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        return functional.mse_loss(scores, labels, reduction=self.reduction)


class MarginRankingLoss(PairwiseLoss):
    """A wrapper around the PyTorch margin ranking loss."""

    synonyms = {"Pairwise Hinge Loss"}

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=dict(type=int, low=0, high=3, q=1),
    )

    def __init__(
        self,
        margin: float = 1.0,
        margin_activation: Callable[[torch.FloatTensor], torch.FloatTensor] = functional.relu,
        reduction: str = 'mean',
    ):
        """Initialize the margin loss instance.

        :param margin:
            The margin by which positive and negative scores should be apart.
        :param margin_activation:
            A margin activation. Defaults to relu, i.e. f(x) = max(0, x), which is the default "margin loss". Using e.g.
            softplus leads to a "soft-margin" formulation, as e.g. discussed here https://arxiv.org/abs/1703.07737
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        """
        super().__init__(reduction=reduction)
        self.margin = margin
        self.margin_activation = margin_activation

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.reduction_operation(self.margin_activation(
            neg_scores[:, :, None] - pos_scores[:, None, :] + self.margin,
        ))


class SoftplusLoss(PointwiseLoss):
    """A loss function for the softplus."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)

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

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=dict(type=int, low=3, high=30, q=3),
        adversarial_temperature=dict(type=float, low=0.5, high=1.0),
    )

    def __init__(self, margin: float = 9.0, adversarial_temperature: float = 1.0, reduction: str = 'mean') -> None:
        """Initialize the NSSA loss.

        :param margin: The loss's margin (also written as gamma in the reference paper)
        :param adversarial_temperature: The negative sampling temperature (also written as alpha in the reference paper)
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.

        .. note:: The default hyperparameters are based the experiments for FB15K-237 in [sun2019]_.
        """
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


def has_mr_loss(model) -> bool:
    """Check if the model has a marging ranking loss."""
    return isinstance(model.loss, MarginRankingLoss)


def has_nssa_loss(model) -> bool:
    """Check if the model has a NSSA loss."""
    return isinstance(model.loss, NSSALoss)
