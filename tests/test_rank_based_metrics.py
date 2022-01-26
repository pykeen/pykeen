"""Tests for rank-based metrics."""
from pykeen.evaluation import rank_based_evaluator
from tests import cases


class ArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for arithmetic mean rank."""

    cls = rank_based_evaluator.ArithmeticMeanRank


class InverseArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse arithmetic mean rank."""

    cls = rank_based_evaluator.InverseArithmeticMeanRank
