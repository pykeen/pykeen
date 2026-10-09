"""Tests for evaluation utilities."""

import unittest
import warnings

import numpy as np
import pytest

from pykeen.metrics.classification import (
    FalseOmissionRate,
    PrevalenceThreshold,
    classification_metric_resolver,
    construct_indicator,
)


def get_true(pos: int = 5, neg: int = 5) -> np.ndarray:
    """Get a true vector."""
    y_true = np.concatenate(
        [
            np.ones(pos),
            np.zeros(neg),
        ]
    )
    assert (np.array([1] * pos + [0] * neg) == y_true).all()
    return y_true


class TestIndicators(unittest.TestCase):
    """Test indicators."""

    def test_indicator(self):
        """Test constructing an indicator."""
        y_score = np.array([5, 6, 7, 8])
        y_true = np.array([1, 0, 0, 1])
        assert construct_indicator(y_score=y_score, y_true=y_true).tolist() == [0, 0, 1, 1]

    def test_indicator_linear_invariant(self):
        """Test that the construction of the indicator is invariant to linear transformations."""
        y_true = get_true()
        for m, b in [
            # (-1, 1),
            (1, 1),
            # (-1, -1),
            (1, -1),
            (5, 3),
            # (-5, -3),
        ]:
            with self.subTest(m=m, b=b):
                y_score = y_true * m + b
                indicator = construct_indicator(y_score=y_score, y_true=y_true)
                assert (indicator == y_true).all(), f"{m}x + {b}"


@pytest.mark.parametrize(("policy", "expected"), [(0, 0.0), (1, 0.5)])
def test_prevalence_threshold_zero_division_policy(policy, expected):
    """Test that the inner rates of the prevalence threshold honor the zero-division policy."""
    metric = PrevalenceThreshold()
    metric.zero_division = policy
    # no positives and no negatives: FPR and TPR are both undefined => both equal to the policy
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert metric.extract_from_confusion_matrix(tn=0, fp=0, fn=0, tp=0) == expected


@pytest.mark.parametrize("key", ["for", "fom"])
def test_false_omission_rate_synonyms(key):
    """Test that the abbreviations resolve to the false omission rate."""
    assert classification_metric_resolver.lookup(key) is FalseOmissionRate
