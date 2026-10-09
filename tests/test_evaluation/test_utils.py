"""Tests for evaluation utilities."""

import unittest

import numpy as np
from sklearn.metrics import precision_score, recall_score

from pykeen.metrics.classification import FowlkesMallowsIndex, construct_indicator


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


class TestFowlkesMallowsIndex(unittest.TestCase):
    """Test the Fowlkes-Mallows index."""

    def test_perfect_prediction(self):
        """A perfect prediction has an index of exactly one (and not more)."""
        assert FowlkesMallowsIndex().extract_from_confusion_matrix(tn=7, fp=0, fn=0, tp=3) == 1.0

    def test_hand_computed(self):
        """Test a non-trivial case: PPV = 2/3, TPR = 2/4, FM = sqrt(1/3)."""
        value = FowlkesMallowsIndex().extract_from_confusion_matrix(tn=5, fp=1, fn=2, tp=2)
        assert abs(value - (1 / 3) ** 0.5) < 1e-12

    def test_matches_sklearn_precision_recall(self):
        """Test against sqrt(precision * recall) computed by sklearn."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        expected = (precision_score(y_true, y_pred) * recall_score(y_true, y_pred)) ** 0.5
        value = FowlkesMallowsIndex().extract_from_confusion_matrix(tn=5, fp=1, fn=2, tp=2)
        assert abs(value - expected) < 1e-12
