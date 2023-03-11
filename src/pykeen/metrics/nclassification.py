# -*- coding: utf-8 -*-

"""Classification metrics."""

# TODO: temporary file during refactoring

from __future__ import annotations

from typing import ClassVar, Protocol, Collection

import numpy
from class_resolver import ClassResolver
from sklearn import metrics

from .utils import Metric, ValueRange

__all__ = [
    "ClassificationMetric",
    "construct_indicator",
    "classification_metric_resolver",
]


class ClassificationFunc(Protocol):
    """A protocol for classification functions."""

    # todo: check typing
    def __call__(self, y_true: numpy.ndarray, y_score: numpy.ndarray, /, **kwargs) -> float:
        """
        Calculate the metric.

        :param y_true: shape: (num_samples,)
            the true label, either 0 or 1.
        :param y_score: shape: (num_samples,)
            the predictions, either as continuous scores, or as binarized prediction
            (depending on the concrete metric at hand).

        :return:
            a scalar metric value
        """
        ...


def construct_indicator(*, y_score: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
    """Construct binary indicators from a list of scores.

    If there are $n$ positively labeled entries in ``y_true``, this function
    assigns the top $n$ highest scores in ``y_score`` as positive and remainder
    as negative.

    .. note ::
        Since the method uses the number of true labels to determine a threshold, the
        results will typically be overly optimistic estimates of the generalization performance.

    .. todo ::
        Add a method which estimates a threshold based on a validation set, and applies this
        threshold for binarization on the test set.

    :param y_score:
        A 1-D array of the score values
    :param y_true:
        A 1-D array of binary values (1 and 0)
    :return:
        A 1-D array of indicator values

    .. seealso::

        This implementation was inspired by
        https://github.com/xptree/NetMF/blob/77286b826c4af149055237cef65e2a500e15631a/predict.py#L25-L33
    """
    # TODO: re-consider threshold
    number_pos = numpy.sum(y_true, dtype=int)
    y_sort = numpy.flip(numpy.argsort(y_score))
    y_pred = numpy.zeros_like(y_true, dtype=int)
    y_pred[y_sort[numpy.arange(number_pos)]] = 1
    return y_pred


class ClassificationMetric(Metric):
    """A base class for classification metrics."""

    #: The function that runs the metric
    func: ClassVar[ClassificationFunc]

    def __call__(self, y_true: numpy.ndarray, y_score: numpy.ndarray, weights: numpy.ndarray | None = None) -> float:
        """Evaluate the metric.

        :param y_true: shape: (num_samples,)
            the true labels, either 0 or 1.
        :param y_score: shape: (num_samples,)
            the predictions, either continuous or binarized.
        :param weights: shape: (num_samples,)
            weights for individual predictions

            .. warning ::
                not all metrics support sample weights - check :attr:`supports_weights` first

        :return:
            the scalar metric value
        """
        if weights is None:
            return self.func(y_true, y_score)
        if not self.supports_weights:
            raise ValueError(
                f"{self.__call__.__qualname__} does not support sample weights but received" f"weights={weights}.",
            )
        return self.func(y_true, y_score, sample_weight=weights)


class BinarizedClassificationMetric(ClassificationMetric):
    """A classification metric which requires binarized predictions instead of scores."""

    binarize: ClassVar[bool] = True

    # docstr-coverage: inherited
    def __call__(self, y_true: numpy.ndarray, y_score: numpy.ndarray) -> float:  # noqa: D102
        return super().__call__(y_true=y_true, y_score=construct_indicator(y_score=y_score, y_true=y_true))


class AveragePrecisionScore(ClassificationMetric):
    """The average precision from prediction scores.

    .. note ::
        this metric is different from the area under the precision-recall curve, which uses
        interpolation and can be too optimistic.

    --
    link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    description: The average precision across different thresholds.
    """

    # TODO: can we directly include sklearn's docstring here?

    name = "Average Precision Score (APS)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("aps", "ap")
    supports_weights: ClassVar[bool] = True
    func = metrics.average_precision_score


classification_metric_resolver: ClassResolver[ClassificationMetric] = ClassResolver.from_subclasses(
    base=ClassificationMetric,
    default=AveragePrecisionScore,
)
