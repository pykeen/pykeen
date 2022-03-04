# -*- coding: utf-8 -*-

"""Tests for evaluation utilities."""

import unittest

import numpy as np

from pykeen.metrics.classification import construct_indicator


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
        self.assertEqual([0, 0, 1, 1], construct_indicator(y_score=y_score, y_true=y_true).tolist())

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
                self.assertTrue((indicator == y_true).all(), msg=f"{m}x + {b}")
