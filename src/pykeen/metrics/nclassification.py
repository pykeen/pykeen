# -*- coding: utf-8 -*-

"""Classification metrics."""

# TODO: temporary file during refactoring

from typing import ClassVar, Protocol

import numpy as np
from class_resolver import ClassResolver

from .utils import Metric

__all__ = [
    "ClassificationMetric",
    "construct_indicator",
    "classification_metric_resolver",
]


class ClassificationFunc(Protocol):
    """A protocol for classification functions."""

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, /, **kwargs) -> float:
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


def construct_indicator(*, y_score: np.ndarray, y_true: np.ndarray) -> np.ndarray:
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
    number_pos = np.sum(y_true, dtype=int)
    y_sort = np.flip(np.argsort(y_score))
    y_pred = np.zeros_like(y_true, dtype=int)
    y_pred[y_sort[np.arange(number_pos)]] = 1
    return y_pred


class ClassificationMetric(Metric):
    """A base class for classification metrics."""

    #: A description of the metric
    description: ClassVar[str]
    #: The function that runs the metric
    func: ClassVar[ClassificationFunc]

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        if self.binarize:
            # TODO: re-consider threshold!
            y_score = construct_indicator(y_score=y_score, y_true=y_true)
        return self.func(y_true, y_score)


class BinarizedClassificationMetric(ClassificationMetric):
    """A classification metric which requires binarized predictions instead of scores."""

    binarize: ClassVar[bool] = True

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return super().__call__(y_true=y_true, y_score=construct_indicator(y_score=y_score, y_true=y_true))


classification_metric_resolver: ClassResolver[ClassificationMetric] = ClassResolver.from_subclasses(
    base=ClassificationMetric,
    # default=PRAUC,
)
