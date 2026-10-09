"""Tests for evaluation utilities."""

import unittest
import warnings

import numpy as np
import pytest

from pykeen.metrics.classification import (
    ConfusionMatrixClassificationMetric,
    TrueNegativeRate,
    TruePositiveRate,
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


@pytest.mark.parametrize(
    "metric_cls",
    [
        cls
        for cls in sorted(classification_metric_resolver.lookup_dict.values(), key=lambda cls: cls.__name__)
        if issubclass(cls, ConfusionMatrixClassificationMetric)
    ],
    ids=lambda cls: cls.__name__,
)
@pytest.mark.parametrize("label", [0, 1])
def test_confusion_matrix_metrics_single_class(metric_cls: type[ConfusionMatrixClassificationMetric], label: int):
    """Test that confusion-matrix metrics do not crash if all labels belong to a single class."""
    y_true = np.full(4, label, dtype=int)
    y_score = np.random.default_rng(42).random(4)
    with warnings.catch_warnings():
        # zero-division warnings are expected here
        warnings.simplefilter("ignore")
        value = metric_cls()(y_true=y_true, y_score=y_score)
    # note: the zero-division policy may yield an int (e.g., 0)
    assert isinstance(value, (int, float))


def test_single_class_values():
    """Test the values for single-class input, where the confusion matrix is known."""
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # all positive: every sample is predicted positive => tp = 4, fn = 0
        assert TruePositiveRate()(y_true=np.ones(4, dtype=int), y_score=y_score) == 1.0
        # all negative: every sample is predicted negative => tn = 4, fp = 0
        assert TrueNegativeRate()(y_true=np.zeros(4, dtype=int), y_score=y_score) == 1.0
