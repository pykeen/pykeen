"""Tests for rank-based metrics."""
import unittest_templates

from pykeen.evaluation import rank_based_evaluator
from tests import cases


class AdjustedArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted arithmetic mean rank."""

    cls = rank_based_evaluator.AdjustedArithmeticMeanRank


class AdjustedArithmeticMeanRankIndexTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted arithmetic mean rank index."""

    cls = rank_based_evaluator.AdjustedArithmeticMeanRankIndex


class ArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for arithmetic mean rank."""

    cls = rank_based_evaluator.ArithmeticMeanRank


class CountTests(cases.RankBasedMetricTestCase):
    """Tests for rank count."""

    cls = rank_based_evaluator.Count


class GeometricMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for geometric mean rank."""

    cls = rank_based_evaluator.GeometricMeanRank


class HarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for harmonic mean rank."""

    cls = rank_based_evaluator.HarmonicMeanRank


class HitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for Hits at k."""

    cls = rank_based_evaluator.HitsAtK


class InverseArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse arithmetic mean rank."""

    cls = rank_based_evaluator.InverseArithmeticMeanRank


class InverseMedianRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse median rank."""

    cls = rank_based_evaluator.InverseMedianRank


class InverseGeometricMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse geometric mean rank."""

    cls = rank_based_evaluator.InverseGeometricMeanRank


class InverseHarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse harmonic mean rank."""

    cls = rank_based_evaluator.InverseHarmonicMeanRank


class MedianAbsoluteDeviationTests(cases.RankBasedMetricTestCase):
    """Tests for MAD."""

    cls = rank_based_evaluator.MedianAbsoluteDeviation


class MedianRankTests(cases.RankBasedMetricTestCase):
    """Tests for median rank."""

    cls = rank_based_evaluator.MedianRank


class StandardDeviationTests(cases.RankBasedMetricTestCase):
    """Tests for rank standard deviation."""

    cls = rank_based_evaluator.StandardDeviation


class VarianceTests(cases.RankBasedMetricTestCase):
    """Tests for rank variance."""

    cls = rank_based_evaluator.Variance


class RankBasedMetricsTest(unittest_templates.MetaTestCase[rank_based_evaluator.RankBasedMetric]):
    """Test for test coverage for rank-based metrics."""

    base_cls = rank_based_evaluator.RankBasedMetric
    base_test = cases.RankBasedMetricTestCase
