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
    Soft Pointwise Hinge           $g(s, l) = \log(1+\exp(\lambda-\hat{l}*s))$
    Pointwise Logistic (softplus)  $g(s, l) = \log(1+\exp(-\hat{l}*s))$
    =============================  ============================================================

For the pointwise logistic and pointwise hinge losses, $\hat{l}$ has been rescaled from $\{0,1\}$ to $\{-1,1\}$.
The sigmoid logistic loss function is defined as $\sigma(z) = \frac{1}{1 + e^{-z}}$.

.. note::

    The pointwise logistic loss can be considered as a special case of the pointwise soft hinge loss
    where $\lambda = 0$.

Batching
~~~~~~~~
The pointwise loss of a set of triples (i.e., a batch) $\mathcal{L}_L: 2^{\mathcal{T}} \rightarrow \mathbb{R}$ is
defined as the arithmetic mean of the pointwise losses over each triple in the subset $\mathcal{B} \in 2^{\mathcal{T}}$:

.. math::

    \mathcal{L}_L(\mathcal{B}) = \frac{1}{|\mathcal{B}|} \sum \limits_{k \in \mathcal{B}} L(k)

Pairwise Loss Functions
-----------------------
A pairwise loss is applied to a pair of triples - a positive and a negative one. It is defined as $L: \mathcal{K}
\times \mathcal{\bar{K}} \rightarrow \mathbb{R}$ and computes a real value for the pair.

All loss functions implemented in PyKEEN induce an auxillary loss function based on the chosen interaction
function $L{*}: \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$ that simply passes the scores through.
Note that $L$ is often used interchangbly with $L^{*}$.

.. math::

    L(k, \bar{k}) = L^{*}(f(k), f(\bar{k}))

Delta Pairwise Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Delta pairwise losses are computed on the differences between the scores of the positive and negative
triples (e.g., $\Delta := f(k) - f(\bar{k})$) with transfer function $g: \mathbb{R} \rightarrow \mathbb{R}$ that take
the form of:

.. math::

    L^{*}(f(k), f(\bar{k})) = g(f(k) - f(\bar{k})) := g(\Delta)

The following table shows delta pairwise loss functions:

.. table::
    :align: center
    :widths: auto

    =========================================  ===========  ======================  ==============================================
    Pairwise Loss                              Activation   Margin                  Formulation
    =========================================  ===========  ======================  ==============================================
    Pairwise Hinge (margin ranking)            ReLU         $\lambda \neq 0$        $g(\Delta) = \max(0, \Delta + \lambda)$
    Soft Pairwise Hinge (soft margin ranking)  softplus     $\lambda \neq 0$        $g(\Delta) = \log(1 + \exp(\Delta + \lambda))$
    Pairwise Logistic                          softplus     $\lambda=0$             $g(\Delta) = \log(1 + \exp(\Delta))$
    =========================================  ===========  ======================  ==============================================

.. note::

    The pairwise logistic loss can be considered as a special case of the pairwise soft hinge loss
    where $\lambda = 0$.

Inseparable Pairwise Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following pairwise loss function use the full generalized form of $L(k, \bar{k}) = \dots$
for their definitions:

.. table::
    :align: center
    :widths: auto

    ==============  ===================================================
    Pairwise Loss   Formulation
    ==============  ===================================================
    Double Loss     $h(\bar{\lambda} + f(\bar{k})) + h(\lambda - f(k))$
    ==============  ===================================================

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
"""  # noqa: E501

import logging
import math
from textwrap import dedent
from typing import Any, ClassVar, Mapping, Optional, Set, Tuple

import torch
from class_resolver import ClassResolver, Hint
from class_resolver.contrib.torch import margin_activation_resolver
from docdata import parse_docdata
from torch import nn
from torch.nn import functional
from torch.nn.modules.loss import _Loss

__all__ = [
    # Base Classes
    "Loss",
    "PointwiseLoss",
    "DeltaPointwiseLoss",
    "MarginPairwiseLoss",
    "PairwiseLoss",
    "SetwiseLoss",
    # Concrete Classes
    "BCEAfterSigmoidLoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "InfoNCELoss",
    "MarginRankingLoss",
    "MSELoss",
    "NSSALoss",
    "SoftplusLoss",
    "SoftPointwiseHingeLoss",
    "PointwiseHingeLoss",
    "DoubleMarginLoss",
    "SoftMarginRankingLoss",
    "PairwiseLogisticLoss",
    # Utils
    "loss_resolver",
]

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_HPO_STRATEGY = dict(type=float, low=0, high=3)


def apply_label_smoothing(
    labels: torch.FloatTensor,
    epsilon: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> torch.FloatTensor:
    """Apply label smoothing to a target tensor.

    Redistributes epsilon probability mass from the true target uniformly to the remaining classes by replacing
        * a hard one by (1 - epsilon)
        * a hard zero by epsilon / (num_classes - 1)

    :param labels:
        The one-hot label tensor.
    :param epsilon:
        The smoothing parameter. Determines how much probability should be transferred from the true class to the
        other classes.
    :param num_classes:
        The number of classes.
    :returns: A smoothed label tensor
    :raises ValueError: if epsilon is negative or if num_classes is None

    ..seealso:
        https://www.deeplearningbook.org/contents/regularization.html, chapter 7.5.1
    """
    if not epsilon:  # either none or zero
        return labels
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be positive, but is {epsilon}")
    if num_classes is None:
        raise ValueError("must pass num_classes to perform label smoothing")

    new_label_true = 1.0 - epsilon
    new_label_false = epsilon / (num_classes - 1)
    return new_label_true * labels + new_label_false * (1.0 - labels)


class UnsupportedLabelSmoothingError(RuntimeError):
    """Raised if a loss does not support label smoothing."""

    def __init__(self, instance: object):
        """Initialize the error."""
        self.instance = instance

    def __str__(self) -> str:
        return f"{self.instance.__class__.__name__} does not support label smoothing."


_REDUCTION_METHODS = dict(
    mean=torch.mean,
    sum=torch.sum,
)


class Loss(_Loss):
    """A loss function."""

    #: synonyms of this loss
    synonyms: ClassVar[Optional[Set[str]]] = None

    #: The default strategy for optimizing the loss's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the loss.

        :param reduction:
            the reduction, cf. `_Loss.__init__`
        """
        super().__init__(reduction=reduction)
        self._reduction_method = _REDUCTION_METHODS[reduction]

    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Process scores from sLCWA training loop.

        :param positive_scores: shape: (batch_size, 1)
            The scores for positive triples.
        :param negative_scores: shape: (batch_size, num_neg_per_pos) or (num_unfiltered_negatives,)
            The scores for the negative triples, either in dense 2D shape, or in case they are already filtered, in
            sparse shape. If they are given in sparse shape, batch_filter needs to be provided, too.
        :param label_smoothing:
            An optional label smoothing parameter.
        :param batch_filter: shape: (batch_size, num_neg_per_pos)
            An optional filter of negative scores which were kept. Given if and only if negative_scores have been
            pre-filtered.
        :param num_entities:
            The number of entities. Only required if label smoothing is enabled.

        :return:
            A scalar loss term.
        """
        # flatten and stack
        positive_scores = positive_scores.view(-1)
        negative_scores = negative_scores.view(-1)
        predictions = torch.cat([positive_scores, negative_scores], dim=0)
        labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])

        # apply label smoothing if necessary.
        labels = apply_label_smoothing(
            labels=labels,
            epsilon=label_smoothing,
            num_classes=num_entities,
        )

        return self(predictions, labels)

    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Process scores from LCWA training loop.

        :param predictions: shape: (batch_size, num_entities)
            The scores.
        :param labels: shape: (batch_size, num_entities)
            The labels.
        :param label_smoothing:
            An optional label smoothing parameter.
        :param num_entities:
            The number of entities.

        :return:
            A scalar loss value.
        """
        # TODO: Do label smoothing only once
        labels = apply_label_smoothing(
            labels=labels,
            epsilon=label_smoothing,
            num_classes=num_entities,
        )
        return self(predictions, labels)


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


@parse_docdata
class BCEWithLogitsLoss(PointwiseLoss):
    r"""The binary cross entropy loss.

    For label function :math:`l:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \{0,1\}` and interaction
    function :math:`f:\mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}`,
    the binary cross entropy loss is defined as:

    .. math::

        L(h, r, t) = -(l(h,r,t) \cdot \log(\sigma(f(h,r,t))) + (1 - l(h,r,t)) \cdot \log(1 - \sigma(f(h,r,t))))

    where represents the logistic sigmoid function

    .. math::

        \sigma(x) = \frac{1}{1 + \exp(-x)}

    .. note::

        The softplus activation function $h_{\text{softplus}}(x) = -\log(\sigma(x))$.

    Thus, the problem is framed as a binary classification problem of triples, where the interaction functions' outputs
    are regarded as logits.

    .. warning::

        This loss is not well-suited for translational distance models because these models produce
        a negative distance as score and cannot produce positive model outputs.

    .. note::

        The related :mod:`torch` module is :class:`torch.nn.BCEWithLogitsLoss`, but it can not be used
        interchangeably in PyKEEN because of the extended functionality implemented in PyKEEN's loss functions.
    ---
    name: Binary cross entropy (with logits)
    """

    synonyms = {"Negative Log Likelihood Loss"}

    # docstr-coverage: inherited
    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return functional.binary_cross_entropy_with_logits(scores, labels, reduction=self.reduction)


@parse_docdata
class MSELoss(PointwiseLoss):
    """The mean squared error loss.

    .. note::

        The related :mod:`torch` module is :class:`torch.nn.MSELoss`, but it can not be used
        interchangeably in PyKEEN because of the extended functionality implemented in PyKEEN's loss functions.
    ---
    name: Mean squared error
    """

    synonyms = {"Mean Square Error Loss", "Mean Squared Error Loss"}

    # docstr-coverage: inherited
    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        assert self.validate_labels(labels=labels)
        return functional.mse_loss(scores, labels, reduction=self.reduction)


class MarginPairwiseLoss(PairwiseLoss):
    r"""The generalized margin ranking loss.

    .. math ::
        L(k, \bar{k}) = g(f(\bar{k}) - f(k) + \lambda)

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    :class:`pykeen.models.TransE` has $f(h,r,t)=\mathbf{e}_h+\mathbf{r}_r-\mathbf{e}_t$), $g(x)$ is an activation
    function like the ReLU or softmax, and $\lambda$ is the margin.
    """

    def __init__(
        self,
        margin: float,
        margin_activation: Hint[nn.Module],
        reduction: str = "mean",
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
        self.margin_activation = margin_activation_resolver.make(margin_activation)

    # docstr-coverage: inherited
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        # prepare for broadcasting, shape: (batch_size, 1, 3)
        positive_scores = positive_scores.unsqueeze(dim=1)

        if batch_filter is not None:
            # negative_scores have already been filtered in the sampler!
            num_neg_per_pos = batch_filter.shape[1]
            positive_scores = positive_scores.repeat(1, num_neg_per_pos, 1)[batch_filter]
            # shape: (nnz,)

        return self(pos_scores=positive_scores, neg_scores=negative_scores)

    # docstr-coverage: inherited
    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        # for LCWA scores, we consider all pairs of positive and negative scores for a single batch element.
        # note: this leads to non-uniform memory requirements for different batches, depending on the total number of
        # positive entries in the labels tensor.

        # This shows how often one row has to be repeated,
        # shape: (batch_num_positives,), if row i has k positive entries, this tensor will have k entries with i
        repeat_rows = (labels == 1).nonzero(as_tuple=False)[:, 0]
        # Create boolean indices for negative labels in the repeated rows, shape: (batch_num_positives, num_entities)
        labels_negative = labels[repeat_rows] == 0
        # Repeat the predictions and filter for negative labels, shape: (batch_num_pos_neg_pairs,)
        negative_scores = predictions[repeat_rows][labels_negative]

        # This tells us how often each true label should be repeated
        repeat_true_labels = (labels[repeat_rows] == 0).nonzero(as_tuple=False)[:, 0]
        # First filter the predictions for true labels and then repeat them based on the repeat vector
        positive_scores = predictions[labels == 1][repeat_true_labels]

        return self(pos_scores=positive_scores, neg_scores=negative_scores)

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the margin loss.

        The scores have to be in broadcastable shape.

        :param pos_scores:
            The positive scores.
        :param neg_scores:
            The negative scores.

        :return:
            A scalar loss term.
        """
        return self._reduction_method(
            self.margin_activation(
                neg_scores - pos_scores + self.margin,
            )
        )


@parse_docdata
class MarginRankingLoss(MarginPairwiseLoss):
    r"""The pairwise hinge loss (i.e., margin ranking loss).

    .. math ::
        L(k, \bar{k}) = \max(0, f(k) - f(\bar{k}) + \lambda)

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    TransE has $f(h,r,t)=h+r-t$), $g(x)=\max(0,x)$ is the ReLU activation function,
    and $\lambda$ is the margin.

    .. seealso::

        MRL is closely related to :class:`pykeen.losses.SoftMarginRankingLoss`, only differing in that this loss
        uses the ReLU activation and :class:`pykeen.losses.SoftMarginRankingLoss` uses the softmax activation. MRL
        is also related to the :class:`pykeen.losses.PairwiseLogisticLoss` as this is a special case of the
        :class:`pykeen.losses.SoftMarginRankingLoss` with no margin.

    .. note::

        The related :mod:`torch` module is :class:`torch.nn.MarginRankingLoss`, but it can not be used
        interchangeably in PyKEEN because of the extended functionality implemented in PyKEEN's loss functions.
    ---
    name: Margin ranking
    """

    synonyms = {"Pairwise Hinge Loss"}

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=DEFAULT_MARGIN_HPO_STRATEGY,
    )

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        r"""Initialize the margin loss instance.

        :param margin:
            The margin by which positive and negative scores should be apart.
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        """
        super().__init__(margin=margin, margin_activation="relu", reduction=reduction)


@parse_docdata
class SoftMarginRankingLoss(MarginPairwiseLoss):
    r"""The soft pairwise hinge loss (i.e., soft margin ranking loss).

    .. math ::
        L(k, \bar{k}) = \log(1 + \exp(f(k) - f(\bar{k}) + \lambda))

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    :class:`pykeen.models.TransE` has $f(h,r,t)=\mathbf{e}_h+\mathbf{r}_r-\mathbf{e}_t$), $g(x)=\log(1 + \exp(x))$
    is the softmax activation function, and $\lambda$ is the margin.

    .. seealso::

        When choosing `margin=0``, this loss becomes equivalent to :class:`pykeen.losses.SoftMarginRankingLoss`.
        It is also closely related to :class:`pykeen.losses.MarginRankingLoss`, only differing in that this loss
        uses the softmax activation and :class:`pykeen.losses.MarginRankingLoss` uses the ReLU activation.
    ---
    name: Soft margin ranking
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=DEFAULT_MARGIN_HPO_STRATEGY,
    )

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        """
        Initialize the loss.

        :param margin:
            the margin, cf. :meth:`MarginPairwiseLoss.__init__`
        :param reduction:
            the reduction, cf. :meth:`MarginPairwiseLoss.__init__`
        """
        super().__init__(margin=margin, margin_activation="softplus", reduction=reduction)


@parse_docdata
class PairwiseLogisticLoss(SoftMarginRankingLoss):
    r"""The pairwise logistic loss.

    .. math ::
        L(k, \bar{k}) = \log(1 + \exp(f(k) - f(\bar{k})))

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    :class:`pykeen.models.TransE` has $f(h,r,t)=\mathbf{e}_h+\mathbf{r}_r-\mathbf{e}_t$), $g(x)=\log(1 + \exp(x))$
    is the softmax activation function.

    .. seealso::

        This loss is equivalent to :class:`pykeen.losses.SoftMarginRankingLoss` where ``margin=0``. It is also
        closely related to :class:`pykeen.losses.MarginRankingLoss` based on the choice of activation function.
    ---
    name: Pairwise logistic
    """

    # Ensures that for this class incompatible hyper-parameter "margin" of superclass is not used
    # within the ablation pipeline.
    hpo_default: ClassVar[Mapping[str, Any]] = dict()

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the loss.

        :param reduction:
            the reduction, cf. :meth:`SoftMarginRankingLoss.__init__`
        """
        super().__init__(margin=0.0, reduction=reduction)


@parse_docdata
class DoubleMarginLoss(PointwiseLoss):
    r"""A limit-based scoring loss, with separate margins for positive and negative elements from [sun2018]_.

    Despite its similarity to the margin-based loss, this loss is quite different to it, since it uses absolute margins
    for positive/negative scores, rather than comparing the difference. Hence, it has a natural decision boundary
    (somewhere between the positive and negative margin), while still resulting in sparse losses with no gradients for
    sufficiently correct examples.

    .. math ::
        L(k, \bar{k}) = g(\bar{\lambda} + \bar{k}) + h(\lambda - k)

    Where $k$ is positive scores, $\bar{k}$ is negative scores, $\lambda$ is the positive margin, $\bar{\lambda}$ is
    the negative margin, and $g$ is an activation function, like the ReLU or softmax.
    ---
    name: Double Margin
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin_positive=dict(type=float, low=-1, high=1),
        offset=dict(type=float, low=0, high=1),
        positive_negative_balance=dict(type=float, low=1.0e-03, high=1.0 - 1.0e-03),
        margin_activation=dict(
            type="categorical",
            choices=margin_activation_resolver.options,
        ),
    )

    @staticmethod
    def resolve_margin(
        positive_margin: Optional[float],
        negative_margin: Optional[float],
        offset: Optional[float],
    ) -> Tuple[float, float]:
        """Resolve margins from multiple methods how to specify them.

        The method supports three combinations:

        - positive_margin & negative_margin.
            This returns the values as-is.
        - negative_margin & offset
            This sets positive_margin = negative_margin + offset
        - positive_margin & offset
            This sets negative_margin = positive_margin - offset

        .. note ::
            Notice that this method does not apply a precedence between the three methods, but requires the remaining
            parameter to be None. This is done to fail fast on ambiguous input rather than delay a failure to a later
            point in time where it might be harder to find its cause.

        :param positive_margin:
            The (absolute) margin for the positive scores. Should be larger than the negative one.
        :param negative_margin:
            The (absolute) margin for the negative scores. Should be smaller than the positive one.
        :param offset:
            The offset between positive and negative margin. Must be non-negative.

        :returns:
            A pair of the positive and negative margin. Guaranteed to fulfil positive_margin >= negative_margin.

        :raises ValueError:
            In case of an invalid combination.
        """
        # 1. positive & negative margin
        if positive_margin is not None and negative_margin is not None and offset is None:
            if negative_margin > positive_margin:
                raise ValueError(
                    f"Positive margin ({positive_margin}) must not be smaller than the negative one "
                    f"({negative_margin}).",
                )
            return positive_margin, negative_margin

        # 2. negative margin & offset
        if negative_margin is not None and offset is not None and positive_margin is None:
            if offset < 0:
                raise ValueError(f"The offset must not be negative, but it is: {offset}")
            return negative_margin + offset, negative_margin

        # 3. positive margin & offset
        if positive_margin is not None and offset is not None and negative_margin is None:
            if offset < 0:
                raise ValueError(f"The offset must not be negative, but it is: {offset}")
            return positive_margin, positive_margin - offset

        raise ValueError(
            dedent(
                f"""\
            Invalid combination of margins and offset:

                positive_margin={positive_margin}
                negative_margin={negative_margin}
                offset={offset}

            Supported are:
                1. positive & negative margin
                2. negative margin & offset
                3. positive margin & offset
        """
            )
        )

    def __init__(
        self,
        *,
        positive_margin: Optional[float] = 1.0,
        negative_margin: Optional[float] = 0.0,
        offset: Optional[float] = None,
        positive_negative_balance: float = 0.5,
        margin_activation: Hint[nn.Module] = "relu",
        reduction: str = "mean",
    ):
        r"""Initialize the double margin loss.

        .. note ::
            There are multiple variants to set the pair of margins. A full documentation is provided in
            :func:`DoubleMarginLoss.resolve_margins`.

        :param positive_margin:
            The (absolute) margin for the positive scores. Should be larger than the negative one.
        :param negative_margin:
            The (absolute) margin for the negative scores. Should be smaller than the positive one.
        :param offset:
            The offset between positive and negative margin. Must be non-negative.
        :param positive_negative_balance:
            The balance between positive and negative term. Must be in (0, 1).
        :param margin_activation:
            A margin activation. Defaults to ``'relu'``, i.e. $h(\Delta) = max(0, \Delta + \lambda)$, which is the
            default "margin loss". Using ``'softplus'`` leads to a "soft-margin" formulation as discussed in
            https://arxiv.org/abs/1703.07737.
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        :raises ValueError: If the positive/negative balance is not within the right range
        """
        super().__init__(reduction=reduction)
        if not (0 <= positive_negative_balance <= 1):
            raise ValueError(
                f"The positive-negative balance weight must be in (0, 1), but is {positive_negative_balance}",
            )
        self.positive_margin, self.negative_margin = self.resolve_margin(
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            offset=offset,
        )
        self.negative_weight = 1.0 - positive_negative_balance
        self.positive_weight = positive_negative_balance
        self.margin_activation = margin_activation_resolver.make(margin_activation)

    # docstr-coverage: inherited
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        # positive term
        if batch_filter is None:
            # implicitly repeat positive scores
            positive_loss = self.margin_activation(self.positive_margin - positive_scores)
            positive_loss = self._reduction_method(positive_loss)
            if self.reduction == "sum":
                positive_loss = positive_loss * negative_scores.shape[1]
            elif self.reduction != "mean":
                raise NotImplementedError(
                    f"There is not implementation for reduction={self.reduction} and filtered negatives",
                )
        else:
            num_neg_per_pos = batch_filter.shape[1]
            positive_scores = positive_scores.unsqueeze(dim=1).repeat(1, num_neg_per_pos, 1)[batch_filter]
            # shape: (nnz,)
            positive_loss = self._reduction_method(self.margin_activation(self.positive_margin - positive_scores))

        # negative term
        # negative_scores have already been filtered in the sampler!
        negative_loss = self._reduction_method(self.margin_activation(self.negative_margin + negative_scores))
        return self.positive_weight * positive_loss + self.negative_weight * negative_loss

    # docstr-coverage: inherited
    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            labels = apply_label_smoothing(
                labels=labels,
                epsilon=label_smoothing,
                num_classes=num_entities,
            )

        return self(predictions=predictions, labels=labels)

    def forward(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the double margin loss.

        The scores have to be in broadcastable shape.

        :param predictions:
            The predicted scores.
        :param labels:
            The labels.

        :return:
            A scalar loss term.
        """
        return self.positive_weight * self._reduction_method(
            labels * self.margin_activation(self.positive_margin - predictions),
        ) + self.negative_weight * self._reduction_method(
            (1.0 - labels) * self.margin_activation(self.negative_margin + predictions),
        )


class DeltaPointwiseLoss(PointwiseLoss):
    r"""A generic class for delta-pointwise losses.

    =============================  ==========  ======================  ========================================================  =============================================
    Pointwise Loss                 Activation  Margin                  Formulation                                               Implementation
    =============================  ==========  ======================  ========================================================  =============================================
    Pointwise Hinge                ReLU        $\lambda \neq 0$        $g(s, l) = \max(0, \lambda -\hat{l}*s)$                   :class:`pykeen.losses.PointwiseHingeLoss`
    Soft Pointwise Hinge           softplus    $\lambda \neq 0$        $g(s, l) = \log(1+\exp(\lambda -\hat{l}*s))$              :class:`pykeen.losses.SoftPointwiseHingeLoss`
    Pointwise Logistic (softplus)  softplus    $\lambda = 0$           $g(s, l) = \log(1+\exp(-\hat{l}*s))$                      :class:`pykeen.losses.SoftplusLoss`
    =============================  ==========  ======================  ========================================================  =============================================
    """  # noqa:E501

    def __init__(
        self,
        margin: Optional[float] = 0.0,
        margin_activation: Hint[nn.Module] = "softplus",
        reduction: str = "mean",
    ) -> None:
        """
        Initialize the loss.

        :param margin:
            the margin, cf. :meth:`PointwiseLoss.__init__`
        :param margin_activation:
            the margin activation, or a hint thereof, cf. `margin_activation_resolver`.
        :param reduction:
            the reduction, cf. :meth:`PointwiseLoss.__init__`
        """
        super().__init__(reduction=reduction)
        self.margin = margin
        self.margin_activation = margin_activation_resolver.make(margin_activation)

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Calculate the loss for the given scores and labels."""
        assert 0.0 <= labels.min() and labels.max() <= 1.0
        # scale labels from [0, 1] to [-1, 1]
        labels = 2 * labels - 1
        loss = self.margin_activation(self.margin - labels * logits)
        loss = self._reduction_method(loss)
        return loss


@parse_docdata
class PointwiseHingeLoss(DeltaPointwiseLoss):
    r"""
    The pointwise hinge loss.

    .. math ::
        g(s,l) = \max(0, \lambda -\hat{l}*s)

    with scores $s$ and labels $l$ that have been rescaled to  $\hat{l} \in \{-1, 1\}$.
    ---
    name: Pointwise Hinge
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=DEFAULT_MARGIN_HPO_STRATEGY,
    )

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        """
        Initialize the loss.

        :param margin:
            the margin, cf. :meth:`DeltaPointwiseLoss.__init__`
        :param reduction:
            the reduction, cf. :meth:`DeltaPointwiseLoss.__init__`
        """
        super().__init__(margin=margin, margin_activation="relu", reduction=reduction)


@parse_docdata
class SoftPointwiseHingeLoss(DeltaPointwiseLoss):
    r"""The soft pointwise hinge loss.

    This loss is appropriate for interaction functions which do not include a bias term,
    and have a limited value range, e.g., distance-based ones like TransE.

    .. seealso::

        When choosing ``margin=0``, this loss becomes equivalent to :class:`pykeen.losses.SoftplusLoss`.
        It is also closely related to :class:`pykeen.losses.PointwiseHingeLoss`, only differing in that this loss
        uses the softmax activation and :class:`pykeen.losses.PointwiseHingeLoss` uses the ReLU activation.
    ---
    name: Soft Pointwise Hinge
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=DEFAULT_MARGIN_HPO_STRATEGY,
    )

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        """
        Initialize the loss.

        :param margin:
            the margin, cf. :meth:`DeltaPointwiseLoss.__init__`
        :param reduction:
            the reduction, cf. :meth:`DeltaPointwiseLoss.__init__`
        """
        super().__init__(margin=margin, margin_activation="softplus", reduction=reduction)


@parse_docdata
class SoftplusLoss(SoftPointwiseHingeLoss):
    r"""The pointwise logistic loss (i.e., softplus loss).

    .. math ::
        g(s, l) = \log(1 + \exp(-\hat{l} \cdot s))

    with scores $s$ and labels $l$ that have been rescaled to $\hat{l} \in \{-1, 1\}$.

    .. seealso::

        This class is a special case of :class:`pykeen.losses.SoftPointwiseHingeLoss` where the margin
        is set to ``margin=0``.
    ---
    name: Softplus
    """

    # Ensures that for this class incompatible hyper-parameter "margin" of superclass is not used
    # within the ablation pipeline.
    hpo_default: ClassVar[Mapping[str, Any]] = dict()

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize the loss.

        :param reduction:
            the reduction, cf. :meth:`SoftPointwiseHingeLoss.__init__`
        """
        super().__init__(margin=0.0, reduction=reduction)


@parse_docdata
class BCEAfterSigmoidLoss(PointwiseLoss):
    """The numerically unstable version of explicit Sigmoid + BCE loss.

    .. note::

        The related :mod:`torch` module is :class:`torch.nn.BCELoss`, but it can not be used
        interchangeably in PyKEEN because of the extended functionality implemented in PyKEEN's loss functions.
    ---
    name: Binary cross entropy (after sigmoid)
    """

    # docstr-coverage: inherited
    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        return functional.binary_cross_entropy(logits.sigmoid(), labels, **kwargs)


def prepare_negative_scores_for_softmax(
    batch_filter: Optional[torch.LongTensor],
    negative_scores: torch.FloatTensor,
    no_inf_rows: bool,
) -> torch.FloatTensor:
    """
    Prepare negative scores for softmax.

    To compute a softmax over negative scores, we may need to invert the filtering procedure
    to get a dense regularly shaped tensor of shape `(batch_size, num_negatives)`.

    :param negative_scores: shape: (batch_size, num_negatives) | (num_batch_negatives,)
        the negative scores, which may have been filtered
    :param batch_filter: shape: (batch_size, num_negatives)
        the binary mask of corresponding to the non-filtered negative scores. If None, no
        filtering did take place, and nothing has to be done.
    :param no_inf_rows:
        whether to avoid `-inf` rows (if a complete row has been filtered)

    :return: shape: (batch_size, num_negatives)
        a dense view of the negative scores, where previously filtered scores have been
        re-filled as -inf.
    """
    if batch_filter is None:
        return negative_scores

    # negative_scores have already been filtered in the sampler!
    # (dense) softmax requires unfiltered scores / masking
    negative_scores_ = torch.zeros_like(batch_filter, dtype=negative_scores.dtype)
    negative_scores_[batch_filter] = negative_scores
    # we need to fill the scores with -inf for all filtered negative examples
    # EXCEPT if all negative samples are filtered (since softmax over only -inf yields nan)
    fill_mask = ~batch_filter
    if no_inf_rows:
        fill_mask = fill_mask & ~(fill_mask.all(dim=1, keepdim=True))
    negative_scores_[fill_mask] = float("-inf")
    # use filled negatives scores
    return negative_scores_


@parse_docdata
class CrossEntropyLoss(SetwiseLoss):
    """The cross entropy loss that evaluates the cross entropy after softmax output.

    .. note::

        The related :mod:`torch` module is :class:`torch.nn.CrossEntropyLoss`, but it can not be used
        interchangeably in PyKEEN because of the extended functionality implemented in PyKEEN's loss functions.
    ---
    name: Cross entropy
    """

    # docstr-coverage: inherited
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # we need dense negative scores => unfilter if necessary
        negative_scores = prepare_negative_scores_for_softmax(
            batch_filter=batch_filter,
            negative_scores=negative_scores,
            # we may have inf rows, since there will be one additional finite positive score per row
            no_inf_rows=False,
        )
        # combine scores: shape: (batch_size, num_negatives + 1)
        scores = torch.cat(
            [
                positive_scores,
                negative_scores,
            ],
            dim=-1,
        )
        # use sparse version of cross entropy
        true_indices = positive_scores.new_zeros(size=positive_scores.shape[:-1], dtype=torch.long)
        # calculate cross entropy loss
        return functional.cross_entropy(
            input=scores,
            target=true_indices,
            label_smoothing=label_smoothing or 0.0,
            reduction=self.reduction,
        )

    # docstr-coverage: inherited
    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # make sure labels form a proper probability distribution
        labels = functional.normalize(labels, p=1, dim=-1)
        # calculate cross entropy loss
        return functional.cross_entropy(
            input=predictions,
            target=labels,
            label_smoothing=label_smoothing or 0.0,
            reduction=self.reduction,
        )


@parse_docdata
class InfoNCELoss(CrossEntropyLoss):
    r"""The InfoNCE loss with additive margin proposed by [wang2022]_.

    This loss is equivalent to :class:`CrossEntropyLoss`, where the scores have been transformed:

    - positive scores are subtracted by the margin `\gamma` and then divided by the temperature `\tau`

        .. math::
            f'(k) = \frac{f(k) - \gamma}{\tau}

    - negative scores are only divided by the temperature `\tau`

        .. math::
            f'(k^-) = \frac{f(k^-)}{\tau}
    ---
    name: InfoNCE loss with additive margin
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=dict(type=float, low=0.01, high=0.10),
        log_adversarial_temperature=dict(type=float, low=-3.0, high=3.0),
    )
    DEFAULT_LOG_ADVERSARIAL_TEMPERATURE: ClassVar[float] = math.log(0.05)

    def __init__(
        self,
        margin: float = 0.02,
        log_adversarial_temperature: float = DEFAULT_LOG_ADVERSARIAL_TEMPERATURE,
        reduction: str = "mean",
    ) -> None:
        r"""Initialize the loss.

        :param margin:
            The loss's margin (also written as $\gamma$ in the reference paper)

            .. note ::
                In the official implementation, the margin parameter only seems to be used during *training*.
                https://github.com/intfloat/SimKGC/blob/4388ebc0c0011fe333bc5a98d0613ab0d1825ddc/models.py#L92-L94

        :param log_adversarial_temperature:
            The logarithm of the negative sampling temperature (also written as $\tau$ in the reference paper).
            We follow the suggested parametrization which ensures positive temperatures for all hyperparameter values.

            .. note ::
                The adversarial temperature is the inverse of the softmax temperature used when computing the weights!
                Its name is only kept for consistency with the nomenclature of [wang2022]_.

            .. note ::
                In the official implementation, the temperature is a *trainable* parameter, cf.
                https://github.com/intfloat/SimKGC/blob/4388ebc0c0011fe333bc5a98d0613ab0d1825ddc/models.py#L31

        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.

        :raises ValueError:
            if the margin is negative
        """
        if margin < 0:
            raise ValueError(f"Cannot have a negative margin: {margin}")
        super().__init__(reduction=reduction)
        self.inverse_softmax_temperature = math.exp(log_adversarial_temperature)
        self.margin = margin

    # docstr-coverage: inherited
    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # determine positive; do not check with == since the labels are floats
        pos_mask = labels > 0.5
        # subtract margin from positive scores
        predictions = predictions - pos_mask.type_as(predictions) * self.margin
        # divide by temperature
        predictions = predictions / self.inverse_softmax_temperature
        return super().process_lcwa_scores(
            predictions=predictions,
            labels=labels,
            label_smoothing=label_smoothing,
            num_entities=num_entities,
        )

    # docstr-coverage: inherited
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # subtract margin from positive scores
        positive_scores = positive_scores - self.margin
        # normalize positive score shape
        if positive_scores.ndim < negative_scores.ndim:
            positive_scores = positive_scores.unsqueeze(dim=-1)
        # divide by temperature
        positive_scores = positive_scores / self.inverse_softmax_temperature
        negative_scores = negative_scores / self.inverse_softmax_temperature
        return super().process_slcwa_scores(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            label_smoothing=label_smoothing,
            batch_filter=batch_filter,
            num_entities=num_entities,
        )


@parse_docdata
class NSSALoss(SetwiseLoss):
    """The self-adversarial negative sampling loss function proposed by [sun2019]_.

    ---
    name: Self-adversarial negative sampling
    """

    synonyms = {"Self-Adversarial Negative Sampling Loss", "Negative Sampling Self-Adversarial Loss"}

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=dict(type=int, low=3, high=30, q=3),
        adversarial_temperature=dict(type=float, low=0.5, high=1.0),
    )

    def __init__(self, margin: float = 9.0, adversarial_temperature: float = 1.0, reduction: str = "mean") -> None:
        """Initialize the NSSA loss.

        :param margin: The loss's margin (also written as gamma in the reference paper)
        :param adversarial_temperature: The negative sampling temperature (also written as alpha in the reference paper)

            .. note ::
                The adversarial temperature is the inverse of the softmax temperature used when computing the weights!
                Its name is only kept for consistency with the nomenclature of [sun2019]_.
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.

        .. note:: The default hyperparameters are based on the experiments for FB15k-237 in [sun2019]_.
        """
        super().__init__(reduction=reduction)
        self.inverse_softmax_temperature = adversarial_temperature
        self.margin = margin

    # docstr-coverage: inherited
    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        # determine positive; do not check with == since the labels are floats
        pos_mask = labels > 0.5

        # compute negative weights (without gradient tracking)
        # clone is necessary since we modify in-place
        weights = predictions.detach().clone()
        weights[pos_mask] = float("-inf")
        weights = weights.mul(self.inverse_softmax_temperature).softmax(dim=1)

        # Split positive and negative scores
        positive_scores = predictions[pos_mask]
        negative_scores = predictions[~pos_mask]

        return self(
            pos_scores=positive_scores,
            neg_scores=negative_scores,
            neg_weights=weights[~pos_mask],
        )

    # docstr-coverage: inherited
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        negative_scores = prepare_negative_scores_for_softmax(
            batch_filter=batch_filter,
            negative_scores=negative_scores,
            # we do not allow full -inf rows, since we compute the softmax over this tensor
            no_inf_rows=True,
        )

        # compute weights (without gradient tracking)
        assert negative_scores.ndimension() == 2
        weights = negative_scores.detach().mul(self.inverse_softmax_temperature).softmax(dim=-1)

        return self(
            pos_scores=positive_scores,
            neg_scores=negative_scores,
            neg_weights=weights,
        )

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
        neg_weights: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Calculate the loss for the given scores.

        :param pos_scores: shape: s_p
            Positive score tensor
        :param neg_scores: shape: s_n
            Negative score tensor
        :param neg_weights: shape: s_n

        :returns: A loss value

        .. seealso:: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py
        """
        # -w * log sigma(-(m + n)) - log sigma (m + p)
        # p >> -m => m + p >> 0 => sigma(m + p) ~= 1 => log sigma(m + p) ~= 0 => -log sigma(m + p) ~= 0
        # p << -m => m + p << 0 => sigma(m + p) ~= 0 => log sigma(m + p) << 0 => -log sigma(m + p) >> 0
        neg_loss = functional.logsigmoid(-neg_scores - self.margin)
        neg_loss = neg_weights * neg_loss
        neg_loss = self._reduction_method(neg_loss)
        pos_loss = functional.logsigmoid(self.margin + pos_scores)
        pos_loss = self._reduction_method(pos_loss)
        loss = -pos_loss - neg_loss

        if self._reduction_method is torch.mean:
            loss = loss / 2.0

        return loss


@parse_docdata
class FocalLoss(PointwiseLoss):
    r"""The focal loss proposed by [lin2018]_.

    It is an adaptation of the (binary) cross entropy loss, which deals better with imbalanced data.
    The implementation is strongly inspired by the implementation in
    :func:`torchvision.ops.sigmoid_focal_loss`, except it is using
    a module rather than the functional form.

    The loss is given as

    .. math ::
        FL(p_t) = -(1 - p_t)^\gamma \log (p_t)

    with :math:`p_t = y \cdot p + (1 - y) \cdot (1 - p)`, where :math:`p` refers to the predicted probability, and `y`
    to the ground truth label in :math:`{0, 1}`.

    Focal loss has some other nice properties, e.g., better calibrated predicted probabilities. See
    [mukhoti2020]_.
    ---
    name: Focal
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the loss module.

        :param gamma: >= 0
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Setting gamma > 0 reduces the
            relative loss for well-classified examples.
            The default value of 2 is taken from [lin2018]_, which report this setting to work best for their
            experiments. However, these experiments where conducted on the task of object classification in images, so
            take it with a grain of salt.
        :param alpha:
            Weighting factor in range (0, 1) to balance positive vs negative examples. alpha is the weight for the
            positive class, i.e., increasing it will let the loss focus more on this class. The weight for the negative
            class is obtained as 1 - alpha.
            [lin2018]_ recommends to either set this to the inverse class frequency, or treat it as a hyper-parameter.
        :param kwargs:
            Additional keyword-based arguments passed to :class:`pykeen.losses.PointwiseLoss`.
        :raises ValueError:
            If alpha is in the wrong range
        """
        super().__init__(**kwargs)
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, but is {gamma}")
        if alpha is not None and not (0 < alpha < 1):
            raise ValueError(f"If alpha is provided, it must be from (0, 1), i.e. the open interval, but it is {alpha}")
        self.alpha = alpha
        self.gamma = gamma

    # docstr-coverage: inherited
    def forward(
        self,
        prediction: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        p = prediction.sigmoid()
        ce_loss = functional.binary_cross_entropy_with_logits(prediction, labels, reduction="none")
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        return self._reduction_method(loss)


loss_resolver: ClassResolver[Loss] = ClassResolver.from_subclasses(
    Loss,
    default=MarginRankingLoss,
    skip={
        PairwiseLoss,
        PointwiseLoss,
        SetwiseLoss,
        DeltaPointwiseLoss,
        MarginPairwiseLoss,
    },
)
for _name, _cls in loss_resolver.lookup_dict.items():
    for _synonym in _cls.synonyms or []:
        loss_resolver.synonyms[_synonym] = _cls
