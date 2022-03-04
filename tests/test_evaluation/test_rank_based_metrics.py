"""Tests for rank-based metrics."""

import unittest_templates

import pykeen.metrics.ranking
from tests import cases


class AdjustedArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted arithmetic mean rank."""

    cls = pykeen.metrics.ranking.AdjustedArithmeticMeanRank


class AdjustedArithmeticMeanRankIndexTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted arithmetic mean rank index."""

    cls = pykeen.metrics.ranking.AdjustedArithmeticMeanRankIndex


class ArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for arithmetic mean rank."""

    cls = pykeen.metrics.ranking.ArithmeticMeanRank


class CountTests(cases.RankBasedMetricTestCase):
    """Tests for rank count."""

    cls = pykeen.metrics.ranking.Count


class GeometricMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for geometric mean rank."""

    cls = pykeen.metrics.ranking.GeometricMeanRank


class HarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for harmonic mean rank."""

    cls = pykeen.metrics.ranking.HarmonicMeanRank


class HitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for Hits at k."""

    cls = pykeen.metrics.ranking.HitsAtK


class InverseArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse arithmetic mean rank."""

    cls = pykeen.metrics.ranking.InverseArithmeticMeanRank


class InverseMedianRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse median rank."""

    cls = pykeen.metrics.ranking.InverseMedianRank


class InverseGeometricMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse geometric mean rank."""

    cls = pykeen.metrics.ranking.InverseGeometricMeanRank


class InverseHarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for inverse harmonic mean rank."""

    cls = pykeen.metrics.ranking.InverseHarmonicMeanRank


class MedianAbsoluteDeviationTests(cases.RankBasedMetricTestCase):
    """Tests for MAD."""

    cls = pykeen.metrics.ranking.MedianAbsoluteDeviation


class MedianRankTests(cases.RankBasedMetricTestCase):
    """Tests for median rank."""

    cls = pykeen.metrics.ranking.MedianRank


class StandardDeviationTests(cases.RankBasedMetricTestCase):
    """Tests for rank standard deviation."""

    cls = pykeen.metrics.ranking.StandardDeviation


class VarianceTests(cases.RankBasedMetricTestCase):
    """Tests for rank variance."""

    cls = pykeen.metrics.ranking.Variance


class RankBasedMetricsTest(unittest_templates.MetaTestCase[pykeen.metrics.ranking.RankBasedMetric]):
    """Test for test coverage for rank-based metrics."""

    base_cls = pykeen.metrics.ranking.RankBasedMetric
    base_test = cases.RankBasedMetricTestCase
