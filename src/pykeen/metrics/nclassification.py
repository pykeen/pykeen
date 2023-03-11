# -*- coding: utf-8 -*-

"""
Classification metrics.

The metrics in this module assume the link prediction setting to be a (binary) classification of individual triples.
"""

# TODO: temporary file during refactoring

from __future__ import annotations

import abc
from typing import ClassVar, Collection, Protocol

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

    # TODO: check how to type properly
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


class ClassificationMetric(Metric, abc.ABC):
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


class BinarizedClassificationMetric(ClassificationMetric, abc.ABC):
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


class ConfusionMatrixClassificationMetric(ClassificationMetric, abc.ABC):
    """A classification metric based on the confusion matrix."""

    binarize: ClassVar[bool] = True

    @abc.abstractmethod
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:
        """
        Calculate the metric from the confusion table.

        :param matrix: shape: (2, 2)
            the confusion table
                [[ TP, FN ]
                 [ FP, TN ]]

        :return:
            the scalar metric
        """
        raise NotImplementedError

    def __call__(self, y_true: numpy.ndarray, y_score: numpy.ndarray, weights: numpy.ndarray | None = None) -> float:
        y_pred = construct_indicator(y_score=y_score, y_true=y_true)
        matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=weights, normalize=None)
        return self.extract_from_confusion_matrix(matrix=matrix)


class TruePositiveRate(ConfusionMatrixClassificationMetric):
    """
    The true positive rate is the probability that the prediction is positive, given the triple is truly positive.

    .. math ::
        TPR = TP / (TP + FN)

    --
    link: https://en.wikipedia.org/wiki/Sensitivity_(test)
    description: The probability that a truly positive triple is predicted positive.
    """

    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("tpr", "sensitivity")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        # TODO: avoid division by zero?
        return matrix[1, 1] / matrix[1, :].sum()


class TrueNegativeRate(ConfusionMatrixClassificationMetric):
    """
    The true negative rate is the probability that the prediction is negative, given the triple is truly negative.

    .. math ::
        TNR = TN / (TN + FP)

    .. warning ::
        most knowledge graph datasets do not have true negatives, i.e., verified false facts, but rather are
        collection of (mostly) true facts, where the missing ones are generally unknown rather than false.

    --
    link: https://en.wikipedia.org/wiki/Specificity_(tests)
    description: The probability that a truly false triple is predicted negative.
    """

    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("tnr", "specificity")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        # TODO: avoid division by zero?
        return matrix[0, 0] / matrix[0, :].sum()


classification_metric_resolver: ClassResolver[ClassificationMetric] = ClassResolver.from_subclasses(
    base=ClassificationMetric,
    default=AveragePrecisionScore,
    skip={BinarizedClassificationMetric, ConfusionMatrixClassificationMetric},
)
