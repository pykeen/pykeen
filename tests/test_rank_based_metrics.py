"""Tests for rank-based metrics."""
import unittest_templates

from pykeen.evaluation import rank_based_evaluator
from tests import cases


class ArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for arithmetic mean rank."""

    cls = rank_based_evaluator.ArithmeticMeanRank


class InverseArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse arithmetic mean rank."""

    cls = rank_based_evaluator.InverseArithmeticMeanRank


class RankBasedMetricsTest(unittest_templates.MetaTestCase[rank_based_evaluator.RankBasedMetric]):
    """Test for test coverage for rank-based metrics."""

    base_cls = rank_based_evaluator.RankBasedMetric
    base_test = cases.RankBasedMetricTestCase
