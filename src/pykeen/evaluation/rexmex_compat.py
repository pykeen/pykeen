# -*- coding: utf-8 -*-

"""Temporary storage of annotations to :mod:`rexmex` functions.

Hopefully https://github.com/AstraZeneca/rexmex/pull/29 will get accepted
so we can rely on first-class annotations of functions.

Periodically, run ``python -m pykeen.evaluation.rexmex_compat`` to be informed of new metrics available
through :mod:`rexmex`.
"""

import inspect

import rexmex.metrics.classification as rmc

from .utils import MetricAnnotator

__all__ = [
    "classifier_annotator",
]

#: Functions with the right signature in the :mod:`rexmex.metrics.classification` that are not themselves metrics
EXCLUDE = {
    rmc.true_positive,
    rmc.true_negative,
    rmc.false_positive,
    rmc.false_negative,
}

#: This dictionary maps from duplicate functions to the canonical function in :mod:`rexmex.metrics.classification`
DUPLICATE_CLASSIFIERS = {
    rmc.miss_rate: rmc.false_negative_rate,
    rmc.fall_out: rmc.false_positive_rate,
    rmc.selectivity: rmc.true_negative_rate,
    rmc.specificity: rmc.true_negative_rate,
    rmc.hit_rate: rmc.true_positive_rate,
    rmc.sensitivity: rmc.true_positive_rate,
    rmc.critical_success_index: rmc.threat_score,
}

classifier_annotator = MetricAnnotator("classification")
classifier_annotator.higher(rmc.true_negative_rate)
classifier_annotator.higher(rmc.true_positive_rate)
classifier_annotator.higher(rmc.positive_predictive_value)
classifier_annotator.higher(rmc.negative_predictive_value)
classifier_annotator.lower(rmc.false_negative_rate)
classifier_annotator.lower(rmc.false_positive_rate)
classifier_annotator.lower(rmc.false_discovery_rate)
classifier_annotator.lower(rmc.false_omission_rate)
classifier_annotator.higher(rmc.positive_likelihood_ratio, lower=0.0, upper=float("inf"))
classifier_annotator.lower(rmc.negative_likelihood_ratio, lower=0.0, upper=float("inf"))
classifier_annotator.lower(rmc.prevalence_threshold)
classifier_annotator.higher(rmc.threat_score)
classifier_annotator.higher(rmc.fowlkes_mallows_index)
classifier_annotator.higher(rmc.informedness)
classifier_annotator.higher(rmc.markedness)
classifier_annotator.higher(rmc.diagnostic_odds_ratio, lower=0.0, upper=float("inf"))
classifier_annotator.higher(rmc.roc_auc_score, name="Area Under the ROC Curve")
classifier_annotator.higher(rmc.accuracy_score, name="Accuracy")
classifier_annotator.higher(rmc.balanced_accuracy_score, name="Balanced Accuracy")
classifier_annotator.higher(rmc.f1_score, name="F1 Score")
classifier_annotator.higher(rmc.precision_score, name="Precision")
classifier_annotator.higher(rmc.recall_score, name="Recall")
classifier_annotator.higher(rmc.average_precision_score, name="Average Precision")
classifier_annotator.higher(
    rmc.matthews_correlation_coefficient,
    lower=-1.0,
    upper=1.0,
    description="A balanced measure applicable even with class imbalance",
)
classifier_annotator.higher(rmc.pr_auc_score, name="Area Under the Precision-Recall Curve")


def _check():
    """Check that all functions in the classification module are annotated."""
    for func in rmc.__dict__.values():
        if not inspect.isfunction(func):
            continue
        parameters = inspect.signature(func).parameters
        if "y_true" not in parameters or "y_score" not in parameters:
            continue
        if func in EXCLUDE:
            if func in classifier_annotator.metrics:
                raise ValueError(f"should not include {func.__name__}")
            continue
        if func in DUPLICATE_CLASSIFIERS:
            if func in classifier_annotator.metrics:
                raise ValueError(f"{func.__name__} is a duplicate of {DUPLICATE_CLASSIFIERS[func].__name__}")
            continue
        if func not in classifier_annotator.metrics:
            raise ValueError(f"missing rexmex classifier: {func.__name__}")


if __name__ == '__main__':
    _check()
