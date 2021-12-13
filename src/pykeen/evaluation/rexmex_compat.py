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
    rmc.precision_score: rmc.positive_predictive_value,
    rmc.recall_score: rmc.true_positive_rate,
}

classifier_annotator = MetricAnnotator("classification")
classifier_annotator.higher(
    rmc.true_negative_rate,
    description="TN / (TN + FP)",
    link="https://en.wikipedia.org/wiki/Specificity_(tests)",
)
classifier_annotator.higher(
    rmc.true_positive_rate, description="TP / (TP + FN)", link="https://en.wikipedia.org/wiki/Sensitivity_(test)"
)
classifier_annotator.higher(
    rmc.positive_predictive_value,
    description="TP / (TP + FP)",
    link="https://en.wikipedia.org/wiki/Positive_predictive_value",
)
classifier_annotator.higher(
    rmc.negative_predictive_value,
    description="TN / (TN + FN)",
    link="https://en.wikipedia.org/wiki/Negative_predictive_value",
)
classifier_annotator.lower(
    rmc.false_negative_rate,
    description="FN / (FN + TP)",
    link="https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates",
)
classifier_annotator.lower(
    rmc.false_positive_rate,
    description="FP / (FP + TN)",
    link="https://en.wikipedia.org/wiki/False_positive_rate",
)
classifier_annotator.lower(
    rmc.false_discovery_rate,
    description="FP / (FP + TP)",
    link="https://en.wikipedia.org/wiki/False_discovery_rate",
)
classifier_annotator.lower(
    rmc.false_omission_rate,
    description="FN / (FN + TN)",
    link="https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values",
)
classifier_annotator.higher(
    rmc.positive_likelihood_ratio,
    lower=0.0,
    upper=float("inf"),
    description="TPR / FPR",
    link="https://en.wikipedia.org/wiki/Positive_likelihood_ratio",
)
classifier_annotator.lower(
    rmc.negative_likelihood_ratio,
    lower=0.0,
    upper=float("inf"),
    description="FNR / TNR",
    link="https://en.wikipedia.org/wiki/Negative_likelihood_ratio",
)
classifier_annotator.lower(
    rmc.prevalence_threshold,
    description="√FPR / (√TPR + √FPR)",
    link="https://en.wikipedia.org/wiki/Prevalence_threshold",
)
classifier_annotator.higher(
    rmc.threat_score,
    description="TP / (TP + FN + FP)",
    link="https://rexmex.readthedocs.io/en/latest/modules/root.html#rexmex.metrics.classification.threat_score",
)
classifier_annotator.higher(
    rmc.fowlkes_mallows_index,
    description="√PPV x √TPR",
    link="https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index",
)
classifier_annotator.higher(
    rmc.informedness, description="TPR + TNR - 1", link="https://en.wikipedia.org/wiki/Informedness"
)
classifier_annotator.higher(
    rmc.markedness, description="PPV + NPV - 1", link="https://en.wikipedia.org/wiki/Markedness"
)
classifier_annotator.higher(
    rmc.diagnostic_odds_ratio,
    lower=0.0,
    upper=float("inf"),
    description="LR+/LR-",
    link="https://en.wikipedia.org/wiki/Diagnostic_odds_ratio",
)
classifier_annotator.higher(
    rmc.roc_auc_score,
    name="AUC-ROC",
    description="Area Under the ROC Curve",
    link="https://en.wikipedia.org/wiki/Receiver_operating_characteristic",
)
classifier_annotator.higher(
    rmc.accuracy_score,
    binarize=True,
    name="Accuracy",
    description="(TP + TN) / (TP + TN + FP + FN)",
    link="https://en.wikipedia.org/wiki/Accuracy",
)
classifier_annotator.higher(
    rmc.balanced_accuracy_score,
    binarize=True,
    name="Balanced Accuracy",
    description="An adjusted version of the accuracy for imbalanced datasets",
    link="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html",
)
classifier_annotator.higher(
    rmc.f1_score,
    name="F1 Score",
    binarize=True,
    description="2TP / (2TP + FP + FN)",
    link="https://en.wikipedia.org/wiki/F1_score",
)
classifier_annotator.higher(
    rmc.average_precision_score,
    name="Average Precision",
    description="A summary statistic over the precision-recall curve",
    link="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
)
classifier_annotator.higher(
    rmc.matthews_correlation_coefficient,
    binarize=True,
    lower=-1.0,
    upper=1.0,
    description="A balanced measure applicable even with class imbalance",
    link="https://en.wikipedia.org/wiki/Phi_coefficient",
)
# TODO there's something wrong with this, so add it later
# classifier_annotator.higher(
#     rmc.pr_auc_score,
#     name="AUC-PR",
#     description="Area Under the Precision-Recall Curve",
#     link="https://rexmex.readthedocs.io/en/latest/modules/root.html#rexmex.metrics.classification.pr_auc_score",
# )


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


if __name__ == "__main__":
    _check()
