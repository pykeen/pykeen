# -*- coding: utf-8 -*-

r"""Loss functions integrated in PyKEEN.

Rather than re-using the built-in loss functions in PyTorch, we have elected to re-implement
some of the code from :mod:`pytorch.nn.modules.loss` in order to encode the three different
links of loss functions accepted by PyKEEN in a class hierarchy. This allows for PyKEEN to more
dynamically handle different kinds of loss functions as well as share code. Further, it gives
more insight to potential users.

Throughout the following explanations of pointwise loss functions, pairwise loss functions, and setwise
loss functions, we will assume the set of entities $\mathcal{E}$, set of relations $\mathcal{R}$, set of possible
triples $\mathcal{T} = \mathcal{E} \times \mathcal{R} \times \mathcal{E}$, set of possible subsets of possible triples
$2^{\mathcal{T}}$ (i.e., the power set of $\mathcal{T}$), set of positive triples $\mathcal{K}$, set of negative
triples $\mathcal{\bar{K}}$, scoring function (e.g., TransE) $f: \mathcal{T} \rightarrow \mathbb{R}$ and labeling
function $l:\mathcal{T} \rightarrow \{0,1\}$ where a value of 1 denotes the triple is positive (i.e., $(h,r,t) \in
\mathcal{K}$) and a value of 0 denotes the triple is negative (i.e., $(h,r,t) \notin \mathcal{K}$).

.. note::

    In most realistic use cases of knowledge graph embedding models, you will have observed a subset of positive
    triples $\mathcal{T_{obs}} \subset \mathcal{K}$ and no observations over negative triples. Depending on the training
    assumption (sLCWA or LCWA), this will mean negative triples are generated in a variety of patterns.

.. note::

    Following the open world assumption (OWA), triples $\mathcal{\bar{K}}$ are better named "not positive" rather
    than negative. This is most relevant for pointwise loss functions. For pairwise and setwise loss functions,
    triples are compared as being more/less positive and the binary classification is not relevant.

Pointwise Loss Functions
------------------------
A pointwise loss is applied to a single triple. It takes the form of $L: \mathcal{T} \rightarrow \mathbb{R}$ and
computes a real-value for the triple given its labeling. Typically, a pointwise loss function takes the form of
$g: \mathbb{R} \times \{0,1\} \rightarrow \mathbb{R}$ based on the scoring function and labeling function.

.. math::

    L(k) = g(f(k), l(k))


Examples
~~~~~~~~
.. table::
    :align: center
    :widths: auto

    =============================  ============================================================
    Pointwise Loss                 Formulation
    =============================  ============================================================
    Square Error                   $g(s, l) = \frac{1}{2}(s - l)^2$
    Binary Cross Entropy           $g(s, l) = -(l*\log (\sigma(s))+(1-l)*(\log (1-\sigma(s))))$
    Pointwise Hinge                $g(s, l) = \max(0, \lambda -\hat{l}*s)$
    Pointwise Logistic (softplus)  $g(s, l) = \log(1+\exp(-\hat{l}*s))$
    =============================  ============================================================

For the pointwise logistic and pointwise hinge losses, $\hat{l}$ has been rescaled from $\{0,1\}$ to $\{-1,1\}$.
The sigmoid logistic loss function is defined as $\sigma(z) = \frac{1}{1 + e^{-z}}$.

Batching
~~~~~~~~
The pointwise loss of a set of triples (i.e., a batch) $\mathcal{L}_L: 2^{\mathcal{T}} \rightarrow \mathbb{R}$ is
defined as the arithmetic mean of the pointwise losses over each triple in the subset $\mathcal{B} \in 2^{\mathcal{T}}$:

.. math::

    \mathcal{L}_L(\mathcal{B}) = \frac{1}{|\mathcal{B}|} \sum \limits_{k \in \mathcal{B}} L(k)

Pairwise Loss Functions
-----------------------
A pairwise loss is applied to a pair of triples - a positive and a negative one. It is defined as $L: \mathcal{K}
\times \mathcal{\bar{K}} \rightarrow \mathbb{R}$ and computes a real value for the pair. Typically,
a pairwise loss is computed as a function $g$ of the difference between the scores of the positive and negative
triples that takes the form $g: \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$.

.. math::

    L(k, \bar{k}) = g(f(k), f(\bar{k}))

Examples
~~~~~~~~
Typically, $g$ takes the following form in which a function $h: \mathbb{R} \rightarrow \mathbb{R}$
is used on the differences in the scores of the positive an the negative triples.

.. math::

    g(f(k), f(\bar{k})) = h(f(k) - f(\bar{k}))

In the following examples of pairwise loss functions, the shorthand is used: $\Delta := f(k) - f(\bar{k})$. The
pairwise logistic loss can be considered as a special case of the soft margin ranking loss where $\lambda = 0$.

.. table::
    :align: center
    :widths: auto

    ===============================  ==============================================
    Pairwise Loss                    Formulation
    ===============================  ==============================================
    Pairwise Hinge (margin ranking)  $h(\Delta) = \max(0, \Delta + \lambda)$
    Soft Margin Ranking              $h(\Delta) = \log(1 + \exp(\Delta + \lambda))$
    Pairwise Logistic                $h(\Delta) = \log(1 + \exp(\Delta))$
    ===============================  ==============================================

Batching
~~~~~~~~
The pairwise loss for a set of pairs of positive/negative triples $\mathcal{L}_L: 2^{\mathcal{K} \times
\mathcal{\bar{K}}} \rightarrow \mathbb{R}$ is defined as the arithmetic mean of the pairwise losses for each pair of
positive and negative triples in the subset $\mathcal{B} \in 2^{\mathcal{K} \times \mathcal{\bar{K}}}$.

.. math::

    \mathcal{L}_L(\mathcal{B}) = \frac{1}{|\mathcal{B}|} \sum \limits_{(k, \bar{k}) \in \mathcal{B}} L(k, \bar{k})

Setwise Loss Functions
----------------------
A setwise loss is applied to a set of triples which can be either positive or negative. It is defined as
$L: 2^{\mathcal{T}} \rightarrow \mathbb{R}$. The two setwise loss functions implemented in PyKEEN,
:class:`pykeen.losses.NSSALoss` and :class:`pykeen.losses.CrossEntropyLoss` are both widely different
in their paradigms, but both share the notion that triples are not strictly positive or negative.

.. math::

    L(k_1, ... k_n) = g(f(k_1), ..., f(k_n))

Batching
~~~~~~~~
The pairwise loss for a set of sets of triples triples $\mathcal{L}_L: 2^{2^{\mathcal{T}}} \rightarrow \mathbb{R}$
is defined as the arithmetic mean of the setwise losses for each set of
triples $\mathcal{b}$ in the subset $\mathcal{B} \in 2^{2^{\mathcal{T}}}$.

.. math::

    \mathcal{L}_L(\mathcal{B}) = \frac{1}{|\mathcal{B}|} \sum \limits_{\mathcal{b} \in \mathcal{B}} L(\mathcal{b})
"""

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
    r"""A module for the binary cross entropy loss.

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

    .. seealso:: :class:`torch.nn.BCEWithLogitsLoss`
    """

    synonyms = {'Negative Log Likelihood Loss'}

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return functional.binary_cross_entropy_with_logits(scores, labels, reduction=self.reduction)


class MSELoss(PointwiseLoss):
    """A module for the mean square error loss.

    .. seealso:: :class:`torch.nn.MSELoss`
    """

    synonyms = {'Mean Square Error Loss', 'Mean Squared Error Loss'}

    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        return functional.mse_loss(scores, labels, reduction=self.reduction)


MARGIN_ACTIVATIONS: Mapping[str, Callable[[torch.FloatTensor], torch.FloatTensor]] = {
    'relu': functional.relu,
    'softplus': functional.softplus,
}


class MarginRankingLoss(PairwiseLoss):
    """A module for the margin ranking loss.

    .. seealso:: :class:`torch.nn.MarginRankingLoss`
    """

    synonyms = {"Pairwise Hinge Loss"}

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=dict(type=int, low=0, high=3, q=1),
    )

    def __init__(
        self,
        margin: float = 1.0,
        margin_activation: Union[str, Callable[[torch.FloatTensor], torch.FloatTensor]] = 'relu',
        reduction: str = 'mean',
    ):
        r"""Initialize the margin loss instance.

        :param margin:
            The margin by which positive and negative scores should be apart.
        :param margin_activation:
            A margin activation. Defaults to ``'relu'``, i.e. $h(\Delta) = max(0, \Delta + \lambda)$, which is the
            default "margin loss". Using ``'softplus'`` leads to a "soft-margin" formulation as discussed in
            https://arxiv.org/abs/1703.07737.
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        """
        super().__init__(reduction=reduction)
        self.margin = margin

        if isinstance(margin_activation, str):
            self.margin_activation = MARGIN_ACTIVATIONS[margin_activation]
        else:
            self.margin_activation = margin_activation

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return self._reduction_method(self.margin_activation(
            neg_scores - pos_scores + self.margin,
        ))


class SoftplusLoss(PointwiseLoss):
    """A module for the softplus loss."""

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
    """A module for the numerically unstable version of explicit Sigmoid + BCE loss.

    .. seealso:: :class:`torch.nn.BCELoss`
    """

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        post_sigmoid = torch.sigmoid(logits)
        return functional.binary_cross_entropy(post_sigmoid, labels, **kwargs)


class CrossEntropyLoss(SetwiseLoss):
    """A module for the cross entopy loss that evaluates the cross entropy after softmax output.

    .. seealso:: :class:`torch.nn.CrossEntropyLoss`
    """

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
