# -*- coding: utf-8 -*-

"""Loss functions integrated in PyKEEN."""
from typing import Any, Callable, Mapping, Set, Type, Union

import torch
from torch import nn
from torch.nn import functional

from .utils import get_all_subclasses, get_cls, normalize_string

__all__ = [
    'Loss',
    'BCELoss',
    'BCEAfterSigmoidLoss',
    'MarginRankingLoss',
    'MSELoss',
    'SoftplusLoss',
    'NSSALoss',
    'CrossEntropyLoss',
    'MarginRankingLoss',
    'MSELoss',
    'BCELoss',
    'losses_hpo_defaults',
    'get_loss_cls',
]

_REDUCTION_METHODS = dict(
    mean=torch.mean,
    sum=torch.sum,
)


class Loss(nn.Module):
    """A loss function."""

    def __init__(
        self,
        reduction: str = 'mean',
    ):
        """
        Initialize the loss module.

        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        """
        super().__init__()
        self.reduction = reduction

    @property
    def reduction_operation(self) -> Callable[[torch.FloatTensor], torch.FloatTensor]:
        """Return the reduction operation."""
        return _REDUCTION_METHODS[self.reduction]


class PointwiseLoss(Loss):
    """Pointwise loss functions compute an independent loss term for each triple-label pair."""

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the loss function.

        :param scores: (batch_size,)
            The individual triple scores.
        :param labels:  (batch_size,)
            The corresponding labels in [0, 1].

        :return:
            A scalar loss value.
        """
        raise NotImplementedError

    @staticmethod
    def validate_labels(labels: torch.FloatTensor) -> bool:
        """Check whether labels are in [0, 1]."""
        return labels.min() >= 0 and labels.max() <= 1


class BCELoss(PointwiseLoss):
    r"""A wrapper around the PyTorch binary cross entropy loss.

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

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        return functional.binary_cross_entropy_with_logits(scores, labels, reduction=self.reduction)


class BCEAfterSigmoidLoss(PointwiseLoss):
    """A loss function which uses the numerically unstable version of explicit Sigmoid + BCE."""

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        return functional.binary_cross_entropy(torch.sigmoid(scores), labels, reduction=self.reduction)


class MSELoss(PointwiseLoss):
    """A wrapper around the PyTorch mean square error loss."""

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        return functional.mse_loss(scores, labels, reduction=self.reduction)


class SoftplusLoss(PointwiseLoss):
    """A loss function using softplus."""

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        # scale labels from [0, 1] to [-1, 1]
        labels = 2 * labels - 1
        loss = functional.softplus((-1) * labels * scores)
        return self.reduction_operation(loss)


class PairwiseLoss(Loss):
    """Pairwise loss functions compare the scores of a positive triple and a negative triple."""

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the loss function.

        If num_positives > 1, all pairs within one batch will be used.

        :param pos_scores: shape: (batch_size, num_positives)
            Scores for positive triples.
        :param neg_scores: shape: (batch_size, num_negatives)
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
        """
        Initialize the margin loss instance.

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
            neg_scores[:, :, None] - pos_scores[:, None, :] + self.margin
        ))


class SetwiseLoss(Loss):
    """Setwise loss functions compare the scores of several triples."""

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

    def __init__(
        self,
        margin: float,
        adversarial_temperature: float,
        reduction: str = 'mean',
    ):
        """
        Initialize the loss module.

        :param margin:
            The margin parameter to use for the base loss (also written as gamma in the reference paper). Negative
            scores should be below -margin, and positive ones above +margin.
        :param adversarial_temperature:
            The softmax temperature to use for computing the weights of negative scores. Smaller values lead to more
            uniform distribution, whereas large values in the limit only consider the largest negative score.
            Called alpha in the reference paper.
            # TODO: The usage of temperature here is inverse to the "normal" usage of softmax temperature.
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
        neg_loss = -self.reduction_operation(neg_score_weights * functional.logsigmoid(-neg_scores - self.margin))
        pos_loss = -self.reduction_operation(functional.logsigmoid(self.margin + pos_scores))
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            loss = loss / 2.

        return loss


_LOSS_SUFFIX = 'Loss'
_LOSSES: Set[Type[Loss]] = get_all_subclasses(base_class=Loss).difference({PointwiseLoss, PairwiseLoss, SetwiseLoss})

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

#: HPO Defaults for losses
losses_hpo_defaults: Mapping[Type[Loss], Mapping[str, Any]] = {
    MarginRankingLoss: dict(
        margin=dict(type=int, low=0, high=3, q=1),
    ),
    NSSALoss: dict(
        margin=dict(type=int, low=3, high=30, q=3),
        adversarial_temperature=dict(type=float, low=0.5, high=1.0),
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
