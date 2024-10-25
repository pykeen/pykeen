"""
Classification metrics.

The metrics in this module assume the link prediction setting to be a (binary) classification of individual triples.

.. note ::
    many metrics in this module use `scikit-learn` under the hood, cf.
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
"""

# see also: https://cran.r-project.org/web/packages/metrica/vignettes/available_metrics_classification.html

from __future__ import annotations

import abc
import math
import warnings
from collections.abc import Collection
from typing import ClassVar, Literal

import numpy
from class_resolver import ClassResolver
from docdata import parse_docdata
from sklearn import metrics

from .utils import Metric, ValueRange

__all__ = [
    "ClassificationMetric",
    "construct_indicator",
    "classification_metric_resolver",
]

ZeroDivisionPolicy = Literal["warn", 0, 1]


def safe_divide(numerator: float, denominator: float, zero_division: ZeroDivisionPolicy = "warn") -> float:
    """
    Perform division and handle divide-by-zero similar to scikit-learn.

    :param numerator:
        the numerator
    :param denominator:
        the denominator
    :param zero_division:
        the zero-division policy; If "warn", act like 0, but warn about the case.

    :return:
        the division result
    """
    # todo: do we need numpy support?
    if denominator != 0:
        return numerator / denominator
    if zero_division == "warn":
        zero_division = 0
        warnings.warn(
            message=f"Division by zero. Result set to {zero_division} according to policy.",
            category=UserWarning,
            stacklevel=2,
        )
    return zero_division


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

        :raises ValueError:
            when weights are provided but the function does not support them.
        """
        if weights is None:
            return self.forward(y_true=y_true, y_score=y_score)
        if not self.supports_weights:
            raise ValueError(
                f"{self.__call__.__qualname__} does not support sample weights but received" f"weights={weights}.",
            )
        return self.forward(y_true=y_true, y_score=y_score, sample_weight=weights)

    @abc.abstractmethod
    def forward(
        self, y_true: numpy.ndarray, y_score: numpy.ndarray, sample_weight: numpy.ndarray | None = None
    ) -> float:
        """
        Calculate the metric.

        :param y_true: shape: (num_samples,)
            the true label, either 0 or 1.
        :param y_score: shape: (num_samples,)
            the predictions, either as continuous scores, or as binarized prediction
            (depending on the concrete metric at hand).
        :param sample_weight: shape: (num_samples,)
            sample weights

        :return:
            a scalar metric value

        # noqa:DAR202
        """


@parse_docdata
class NumScores(ClassificationMetric):
    """
    The number of scores.

    Lower numbers may indicate unreliable results.
    ---
    description: The number of scores.
    link: https://pykeen.readthedocs.io/en/stable/reference/evaluation.html
    """

    name: ClassVar[str] = "Number of Scores"
    value_range: ClassVar[ValueRange] = ValueRange(lower=0, lower_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("score_count",)

    # docstr-coverage: inherited
    def forward(self, y_true: numpy.ndarray, y_score: numpy.ndarray, weights: numpy.ndarray | None = None) -> float:  # noqa: D102
        return y_score.size


class BinarizedClassificationMetric(ClassificationMetric, abc.ABC):
    """A classification metric which requires binarized predictions instead of scores."""

    binarize: ClassVar[bool] = True

    # docstr-coverage: inherited
    def __call__(self, y_true: numpy.ndarray, y_score: numpy.ndarray, weights: numpy.ndarray | None = None) -> float:  # noqa: D102
        return super().__call__(
            y_true=y_true, y_score=construct_indicator(y_score=y_score, y_true=y_true), weights=weights
        )


@parse_docdata
class BalancedAccuracyScore(BinarizedClassificationMetric):
    """
    The average of recall obtained on each class.

    ---
    link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    description: The average of recall obtained on each class.
    """

    name = "Balanced Accuracy Score"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("b-acc", "bas")
    supports_weights: ClassVar[bool] = True

    # docstr-coverage: inherited
    def forward(
        self, y_true: numpy.ndarray, y_score: numpy.ndarray, sample_weight: numpy.ndarray | None = None
    ) -> float:  # noqa: D102
        return float(metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_score, sample_weight=sample_weight))


@parse_docdata
class AveragePrecisionScore(ClassificationMetric):
    """The average precision from prediction scores.

    .. note ::
        this metric is different from the area under the precision-recall curve, which uses
        interpolation and can be too optimistic.

    ---
    link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    description: The average precision across different thresholds.
    """

    # TODO: can we directly include sklearn's docstring here?

    name = "Average Precision Score"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("aps", "ap")
    supports_weights: ClassVar[bool] = True

    # docstr-coverage: inherited
    def forward(
        self, y_true: numpy.ndarray, y_score: numpy.ndarray, sample_weight: numpy.ndarray | None = None
    ) -> float:  # noqa: D102
        return float(metrics.average_precision_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight))


@parse_docdata
class AreaUnderTheReceiverOperatingCharacteristicCurve(ClassificationMetric):
    """
    The area under the receiver operating characteristic curve.

    ---
    link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    description: The area under the receiver operating characteristic curve.
    """

    name = "Area Under The Receiver Operating Characteristic Curve"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("roc-auc",)
    supports_weights: ClassVar[bool] = True

    # docstr-coverage: inherited
    def forward(
        self, y_true: numpy.ndarray, y_score: numpy.ndarray, sample_weight: numpy.ndarray | None = None
    ) -> float:  # noqa: D102
        return float(metrics.roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight))


class ConfusionMatrixClassificationMetric(ClassificationMetric, abc.ABC):
    """A classification metric based on the confusion matrix."""

    binarize: ClassVar[bool] = True
    zero_division: ZeroDivisionPolicy = "warn"

    @abc.abstractmethod
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:
        """
        Calculate the metric from the confusion table.

        :param matrix: shape: (2, 2)
            the confusion table of the form::

                [[ TP, FN ]
                 [ FP, TN ]]

        :return:
            the scalar metric

        # noqa: DAR202
        """
        # todo: it would make sense to have a separate evaluator which constructs the confusion matrix only once

    # docstr-coverage: inherited
    def forward(self, y_true: numpy.ndarray, y_score: numpy.ndarray, weights: numpy.ndarray | None = None) -> float:  # noqa: D102
        y_pred = construct_indicator(y_score=y_score, y_true=y_true)
        matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=weights, normalize=None)
        return self.extract_from_confusion_matrix(matrix=matrix)


@parse_docdata
class TruePositiveRate(ConfusionMatrixClassificationMetric):
    """
    The true positive rate is the probability that the prediction is positive, given the triple is truly positive.

    .. math ::
        TPR = TP / (TP + FN)

    ---
    link: https://en.wikipedia.org/wiki/Sensitivity_(test)
    description: The probability that a truly positive triple is predicted positive.
    """

    name = "True Positive Rate"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("tpr", "sensitivity", "recall", "hit rate")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[1, 1].item(), denominator=matrix[1, :].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class TrueNegativeRate(ConfusionMatrixClassificationMetric):
    """
    The true negative rate is the probability that the prediction is negative, given the triple is truly negative.

    .. math ::
        TNR = TN / (TN + FP)

    .. warning ::
        most knowledge graph datasets do not have true negatives, i.e., verified false facts, but rather are
        collection of (mostly) true facts, where the missing ones are generally unknown rather than false.

    ---
    link: https://en.wikipedia.org/wiki/Specificity_(tests)
    description: The probability that a truly false triple is predicted negative.
    """

    name = "True Negative Rate"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("tnr", "specificity", "selectivity")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[0, 0].item(), denominator=matrix[0, :].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class FalsePositiveRate(ConfusionMatrixClassificationMetric):
    """
    The false positive rate is the probability that the prediction is positive, given the triple is truly negative.

    .. math ::
        FPR = FP / (FP + TN)

    ---
    link: https://en.wikipedia.org/wiki/False_positive_rate
    description: The probability that a truly negative triple is predicted positive.
    """

    name = "False Positive Rate"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("fpr", "fall-out", "false alarm ratio")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[1, 0].item(), denominator=matrix[1, :].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class FalseNegativeRate(ConfusionMatrixClassificationMetric):
    """
    The false negative rate is the probability that the prediction is negative, given the triple is truly positive.

    .. math ::
        FNR = FN / (FN + TP)

    ---
    link: https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates
    description: The probability that a truly positive triple is predicted negative.
    """

    name = "False Negative Rate"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("fnr", "miss-rate")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[0, 1].item(), denominator=matrix[0, :].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class PositivePredictiveValue(ConfusionMatrixClassificationMetric):
    """
    The positive predictive value is the proportion of predicted positives which are true positive.

    .. math ::
        PPV = TP / (TP + FP)

    ---
    link: https://en.wikipedia.org/wiki/Positive_predictive_value
    description: The proportion of predicted positives which are true positive.
    """

    name = "Positive Predictive Value"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("ppv",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[0, 0].item(), denominator=matrix[:, 0].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class NegativePredictiveValue(ConfusionMatrixClassificationMetric):
    """
    The negative predictive value is the proportion of predicted negatives which are true negative.

    .. math ::
        NPV = TN / (TN + FN)

    ---
    link: https://en.wikipedia.org/wiki/Negative_predictive_value
    description: The proportion of predicted negatives which are true negatives.
    """

    name = "Negative Predictive Value"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("npv",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[1, 1].item(), denominator=matrix[:, 1].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class FalseDiscoveryRate(ConfusionMatrixClassificationMetric):
    """
    The false discovery rate is the proportion of predicted negatives which are true positive.

    .. math ::
        FDR = FP / (FP + TP)

    ---
    link: https://en.wikipedia.org/wiki/False_discovery_rate
    description: The proportion of predicted negatives which are true positive.
    """

    name = "False Discovery Rate"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("fdr",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[1, 0].item(), denominator=matrix[:, 0].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class FalseOmissionRate(ConfusionMatrixClassificationMetric):
    """
    The false omission rate is the proportion of predicted positives which are true negative.

    .. math ::
        FOR = FN / (FN + TN)

    ---
    link: https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values
    description: The proportion of predicted positives which are true negative.
    """

    name = "False Omission Rate"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("fom",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[0, 1].item(), denominator=matrix[:, 1].sum().item(), zero_division=self.zero_division
        )


@parse_docdata
class PositiveLikelihoodRatio(ConfusionMatrixClassificationMetric):
    r"""
    The positive likelihood ratio is the ratio of true positive rate to false positive rate.

    .. math ::
        LR+ = TPR / FPR = \frac{TP / (TP + FN)}{FP / (FP + TN)} = \frac{TP \cdot (FP + TN)}{FP \cdot (TP + FN)}

    ---
    link: https://en.wikipedia.org/wiki/Positive_likelihood_ratio
    description: The ratio of true positive rate to false positive rate.
    """

    name = "Positive Likelihood Ratio"
    value_range = ValueRange(lower=0, lower_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("lr+",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=(matrix[0, 0] * matrix[1, :].sum()).item(),
            denominator=(matrix[1, 0] * matrix[0, :].sum()).item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class NegativeLikelihoodRatio(ConfusionMatrixClassificationMetric):
    r"""
    The negative likelihood ratio is the ratio of false negative rate to true negative rate.

    .. math ::
        LR- = FNR / TNR = \frac{FN / (TP + FN)}{TN / (FP + TN)} = \frac{FN \cdot (FP + TN)}{TN \cdot (TP + FN)}

    ---
    link: https://en.wikipedia.org/wiki/Negative_likelihood_ratio
    description: The ratio of false positive rate to true positive rate.
    """

    name = "Negative Likelihood Ratio"
    value_range = ValueRange(lower=0, lower_inclusive=True)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("lr-",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=(matrix[0, 1] * matrix[0, :].sum()).item(),
            denominator=(matrix[1, 1] * matrix[1, :].sum()).item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class DiagnosticOddsRatio(ConfusionMatrixClassificationMetric):
    r"""
    The ratio of positive and negative likelihood ratio.

    .. math ::
        DOR = \frac{LR+}{LR-} = \frac{TP \cdot TN}{FP \cdot FN}

    ---
    link: https://en.wikipedia.org/wiki/Diagnostic_odds_ratio
    description: The ratio of positive and negative likelihood ratio.
    """

    name = "Diagnostic Odds Ratio"
    value_range = ValueRange(lower=0, lower_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("dor",)

    # todo: https://en.wikipedia.org/wiki/Diagnostic_odds_ratio#Confidence_interval

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=(matrix[0, 0] * matrix[1, 1]).item(),
            denominator=(matrix[0, 1] * matrix[1, 0]).item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class Accuracy(ConfusionMatrixClassificationMetric):
    r"""
    The ratio of the number of correct classifications to the total number.

    .. math ::
        ACC = (TP + TN) / (TP + TN + FP + FN)

    ---
    link: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers#Single_metrics
    description: The ratio of the number of correct classifications to the total number.
    """

    name = "Accuracy"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("acc", "fraction correct", "fc")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=(matrix[0, 0] * matrix[1, 1]).item(),
            denominator=matrix.sum().item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class F1Score(ConfusionMatrixClassificationMetric):
    r"""
    The harmonic mean of precision and recall.

    .. math ::
        F1 = 2TP / (2TP + FP + FN)

    ---
    link: https://en.wikipedia.org/wiki/F1_score
    description: The harmonic mean of precision and recall.
    """

    name = "F1 Score"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("f1",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=2 * matrix[0, 0].item(),
            denominator=(2 * matrix[0, 0] + matrix[0, 1] + matrix[1, 0]).item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class PrevalenceThreshold(ConfusionMatrixClassificationMetric):
    r"""
    The prevalence threshold.

    .. math ::
        PT = √FPR / (√TPR + √FPR)

    ---
    link: https://en.wikipedia.org/wiki/Prevalence_threshold
    description: The prevalence threshold.
    """

    # todo: improve doc
    name = "Prevalence Threshold"
    value_range = ValueRange(lower=0, lower_inclusive=True)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("pt",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        fpr = FalsePositiveRate().extract_from_confusion_matrix(matrix=matrix)
        tpr = TruePositiveRate().extract_from_confusion_matrix(matrix=matrix)
        return safe_divide(
            numerator=numpy.sqrt(fpr).item(),
            denominator=(numpy.sqrt(fpr) + numpy.sqrt(tpr)).item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class ThreatScore(ConfusionMatrixClassificationMetric):
    r"""
    The threat score.

    .. math ::
        TS = TP / (TP + FN + FP)

    ---
    link: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    description: The harmonic mean of precision and recall.
    """

    name = "Threat Score"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("ts", "critical success index", "csi", "jaccard index")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=matrix[0, 0].item(),
            denominator=(matrix[0, 0] + matrix[0, 1] + matrix[1, 0]).item(),
            zero_division=self.zero_division,
        )


@parse_docdata
class FowlkesMallowsIndex(ConfusionMatrixClassificationMetric):
    r"""
    The Fowlkes Mallows index.

    .. math ::
        FM = \sqrt{\frac{TP^2}{(2TP + FP + FN)}}

    ---
    link: https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
    description: The Fowlkes Mallows index.
    """

    name = "Fowlkes Mallows Index"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("fm", "fmi")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return math.sqrt(
            safe_divide(
                numerator=matrix[0, 0].item() ** 2,
                denominator=(2 * matrix[0, 0] + matrix[0, 1] + matrix[1, 0]).item(),
                zero_division=self.zero_division,
            )
        )


@parse_docdata
class Informedness(ConfusionMatrixClassificationMetric):
    r"""
    The informedness metric.

    .. math ::
        YI = TPR + TNR - 1 = TP / (TP + FN) + TN / (TN + FP) - 1

    ---
    link: https://en.wikipedia.org/wiki/Informedness
    description: The informedness metric.
    """

    name = "Informedness"
    value_range = ValueRange(lower=-1, lower_inclusive=True, upper=1, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("Youden's J", "Youden's Index", "yi")

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return (
            safe_divide(
                numerator=matrix[1, 1].item(), denominator=matrix[1, :].sum().item(), zero_division=self.zero_division
            )
            + safe_divide(
                numerator=matrix[0, 0].item(), denominator=matrix[0, :].sum().item(), zero_division=self.zero_division
            )
            - 1
        )


@parse_docdata
class MatthewsCorrelationCoefficient(ConfusionMatrixClassificationMetric):
    r"""
    The Matthews Correlation Coefficient (MCC).

    A balanced measure applicable even with class imbalance.

    .. math ::
        MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    ---
    link: https://en.wikipedia.org/wiki/Phi_coefficient
    description: The Matthews Correlation Coefficient (MCC).
    """

    name = "Matthews Correlation Coefficient"
    value_range: ClassVar[ValueRange] = ValueRange(lower=-1, upper=1, lower_inclusive=True, upper_inclusive=True)
    increasing: ClassVar[bool] = True
    synonyms: ClassVar[Collection[str]] = ("mcc",)

    # docstr-coverage: inherited
    def extract_from_confusion_matrix(self, matrix: numpy.ndarray) -> float:  # noqa: D102
        return safe_divide(
            numerator=(matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]).item(),
            denominator=(matrix.sum(axis=1).prod() * matrix.sum(axis=0).prod()).item(),
            zero_division=self.zero_division,
        )


#: A resolver for classification metrics
classification_metric_resolver: ClassResolver[ClassificationMetric] = ClassResolver.from_subclasses(
    base=ClassificationMetric,
    default=AveragePrecisionScore,
    skip={BinarizedClassificationMetric, ConfusionMatrixClassificationMetric},
)
