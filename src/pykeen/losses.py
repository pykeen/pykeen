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

from typing import Any, ClassVar, Mapping, Optional, Set

import torch
from class_resolver import Hint, Resolver
from torch import nn
from torch.nn import functional
from torch.nn.modules.loss import _Loss

__all__ = [
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
    # Utils
    'loss_resolver',
]


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
    :raises ValueError: if episilon is negative or if num_classes is None

    ..seealso:
        https://www.deeplearningbook.org/contents/regularization.html, chapter 7.5.1
    """
    if not epsilon:  # either none or zero
        return labels
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be positive, but is {epsilon}")
    if num_classes is None:
        raise ValueError("must pass num_classes to perform label smoothing")

    new_label_true = (1.0 - epsilon)
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

    synonyms: ClassVar[Optional[Set[str]]] = None

    #: The default strategy for optimizing the loss's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
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


margin_activation_resolver = Resolver(
    classes={
        nn.ReLU,
        nn.Softplus,
    },
    base=nn.Module,  # type: ignore
    synonyms=dict(
        hard=nn.ReLU,
        soft=nn.Softplus,
    ),
)


class MarginRankingLoss(PairwiseLoss):
    r"""A module for the margin ranking loss.

    .. math ::
        L(score^+, score^-) = activation(score^- - score^+ + margin)

    .. seealso:: :class:`torch.nn.MarginRankingLoss`
    """

    synonyms = {"Pairwise Hinge Loss"}

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=dict(type=int, low=0, high=3, q=1),
        margin_activation=dict(
            type='categorical',
            choices=margin_activation_resolver.options,
        ),
    )

    def __init__(
        self,
        margin: float = 1.0,
        margin_activation: Hint[nn.Module] = 'relu',
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
        self.margin_activation = margin_activation_resolver.make(margin_activation)

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
        return self._reduction_method(self.margin_activation(
            neg_scores - pos_scores + self.margin,
        ))


class SoftplusLoss(PointwiseLoss):
    r"""
    A module for the softplus loss.

    .. math ::
        L(score, label) = softplus(- label \cdot score)

    with $label \in \{-1, 1\}$.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.softplus = nn.Softplus()

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
        return functional.binary_cross_entropy(logits.sigmoid(), labels, **kwargs)


class CrossEntropyLoss(SetwiseLoss):
    """A module for the cross entropy loss that evaluates the cross entropy after softmax output.

    .. seealso:: :class:`torch.nn.CrossEntropyLoss`
    """

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
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

        pos_mask = labels == 1

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

        if batch_filter is not None:
            # negative_scores have already been filtered in the sampler!
            # (dense) softmax requires unfiltered scores / masking
            negative_scores_ = torch.zeros_like(batch_filter, dtype=positive_scores.dtype)
            negative_scores_[batch_filter] = negative_scores
            # we need to fill the scores with -inf for all filtered negative examples
            # EXCEPT if all negative samples are filtered (since softmax over only -inf yields nan)
            fill_mask = ~batch_filter
            fill_mask = fill_mask & ~(fill_mask.all(dim=1, keepdim=True))
            negative_scores_[fill_mask] = float("-inf")
            # use filled negatives scores
            negative_scores = negative_scores_

        # compute weights (without gradient tracking)
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
            loss = loss / 2.

        return loss


loss_resolver = Resolver.from_subclasses(
    Loss,
    default=MarginRankingLoss,
    skip={
        PairwiseLoss,
        PointwiseLoss,
        SetwiseLoss,
    },
)
for _name, _cls in loss_resolver.lookup_dict.items():
    for _synonym in _cls.synonyms or []:
        loss_resolver.synonyms[_synonym] = _cls
