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


class ZInverseHarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted MRR."""

    cls = pykeen.metrics.ranking.ZInverseHarmonicMeanRank

    def test_adjustment(self):
        """Check that the adjustment happened."""
        base_instance = pykeen.metrics.ranking.InverseHarmonicMeanRank()
        self.assertNotEqual(
            self.instance(ranks=self.ranks, num_candidates=self.num_candidates),
            base_instance(ranks=self.ranks, num_candidates=self.num_candidates),
        )


class AdjustedHitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted hits at k."""

    cls = pykeen.metrics.ranking.AdjustedHitsAtK


class AdjustedInverseHarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted MRR."""

    cls = pykeen.metrics.ranking.AdjustedInverseHarmonicMeanRank


class ArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for arithmetic mean rank."""

    cls = pykeen.metrics.ranking.ArithmeticMeanRank
    check_expectation = True
    check_variance = True


class ZArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for z-scored arithmetic mean rank."""

    cls = pykeen.metrics.ranking.ZArithmeticMeanRank

    def test_adjustment(self):
        """Check that the adjustment happened."""
        base_instance = pykeen.metrics.ranking.ArithmeticMeanRank()
        self.assertNotEqual(
            self.instance(ranks=self.ranks, num_candidates=self.num_candidates),
            -base_instance(ranks=self.ranks, num_candidates=self.num_candidates),
        )


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
    check_expectation = True
    check_variance = True


class ZHitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for z-scored hits at k."""

    cls = pykeen.metrics.ranking.ZHitsAtK

    def test_adjustment(self):
        """Check that the adjustment happened."""
        base_instance = pykeen.metrics.ranking.HitsAtK()
        self.assertNotEqual(
            self.instance(ranks=self.ranks, num_candidates=self.num_candidates),
            base_instance(ranks=self.ranks, num_candidates=self.num_candidates),
        )


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
    check_expectation = True
    check_variance = True


class MedianAbsoluteDeviationTests(cases.RankBasedMetricTestCase):
    """Tests for MAD."""

    cls = pykeen.metrics.ranking.MedianAbsoluteDeviation


class MedianRankTests(cases.RankBasedMetricTestCase):
    """Tests for median rank."""

    cls = pykeen.metrics.ranking.MedianRank
    check_expectation = True


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
    skip_cls = {
        pykeen.metrics.ranking.BaseZMixin,
        pykeen.metrics.ranking.IncreasingZMixin,
        pykeen.metrics.ranking.DecreasingZMixin,
        pykeen.metrics.ranking.ExpectationNormalizedMixin,
        pykeen.metrics.ranking.ReindexMixin,
    }
