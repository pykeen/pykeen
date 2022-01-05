# -*- coding: utf-8 -*-

"""Extra annotations on :mod:`rexmex` functions."""

from typing import Union

import rexmex.metrics.classification as rmc

__all__ = [
    "interval",
    "DUPLICATE_CLASSIFIERS",
    "EXCLUDE_CLASSIFIERS",
]


def interval(func) -> str:
    """Get the math notation for the range of this metric."""
    left = "[" if func.lower_inclusive else "("
    right = "]" if func.upper_inclusive else ")"
    lower: Union[int, float]
    upper: Union[int, float]
    try:
        lower = int(func.lower)
    except OverflowError:
        lower = func.lower
        left = "("
    try:
        upper = int(func.upper)
    except OverflowError:
        upper = func.upper
        right = ")"
    return f"{left}{lower}, {upper}{right}"


#: Functions with the right signature in the :mod:`rexmex.metrics.classification` that are not themselves metrics
EXCLUDE_CLASSIFIERS = {
    rmc.pr_auc_score,
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
