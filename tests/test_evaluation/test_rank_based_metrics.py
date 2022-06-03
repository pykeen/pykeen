"""Tests for rank-based metrics."""
import unittest
from typing import Callable, Optional

import numpy
import numpy as np
import unittest_templates
from scipy.stats import bootstrap

import pykeen.metrics.ranking
from pykeen.metrics.ranking import generalized_harmonic_numbers, harmonic_variances
from pykeen.metrics.utils import (
    stable_product,
    weighted_harmonic_mean,
    weighted_mean_expectation,
    weighted_mean_variance,
    weighted_median,
)
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


class AdjustedHitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted hits at k."""

    cls = pykeen.metrics.ranking.AdjustedHitsAtK


class AdjustedInverseHarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted MRR."""

    cls = pykeen.metrics.ranking.AdjustedInverseHarmonicMeanRank


class ArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for arithmetic mean rank."""

    cls = pykeen.metrics.ranking.ArithmeticMeanRank


class ZArithmeticMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for z-scored arithmetic mean rank."""

    cls = pykeen.metrics.ranking.ZArithmeticMeanRank


class CountTests(cases.RankBasedMetricTestCase):
    """Tests for rank count."""

    cls = pykeen.metrics.ranking.Count


class GeometricMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for geometric mean rank."""

    cls = pykeen.metrics.ranking.GeometricMeanRank


class AdjustedGeometricMeanRankIndexTests(cases.RankBasedMetricTestCase):
    """Tests for adjusted geometric mean rank index."""

    cls = pykeen.metrics.ranking.AdjustedGeometricMeanRankIndex


class ZGeometricMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for z-geometric mean rank."""

    cls = pykeen.metrics.ranking.ZGeometricMeanRank


class HarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for harmonic mean rank."""

    cls = pykeen.metrics.ranking.HarmonicMeanRank


class HitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for Hits at k."""

    cls = pykeen.metrics.ranking.HitsAtK


class ZHitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for z-scored hits at k."""

    cls = pykeen.metrics.ranking.ZHitsAtK


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
    skip_cls = {
        pykeen.metrics.ranking.ExpectationNormalizedMetric,
        pykeen.metrics.ranking.ReindexedMetric,
        pykeen.metrics.ranking.ZMetric,
        pykeen.metrics.ranking.DerivedRankBasedMetric,
    }


class BaseExpectationTests(unittest.TestCase):
    """Verification of expectation and variance of individual ranks."""

    n: int = 1_000

    def setUp(self) -> None:
        """Prepare ranks."""
        self.ranks = numpy.arange(1, self.n + 1).astype(float)

    def test_rank_mean(self):
        """Verify expectation of individual ranks."""
        # expectation = (1 + n) / 2
        mean = self.ranks.mean()
        numpy.testing.assert_allclose(mean, 0.5 * (1 + self.n))

    def test_rank_var(self):
        """Verify variance of individual ranks."""
        # variance = (n**2 - 1) / 12
        variance = self.ranks.var()
        numpy.testing.assert_allclose(variance, (self.n**2 - 1) / 12.0)

    def test_inverse_rank_mean(self):
        """Verify the expectation of the inverse rank."""
        mean = np.reciprocal(self.ranks).mean()
        numpy.testing.assert_allclose(mean, generalized_harmonic_numbers(n=self.n, p=-1)[-1] / self.n)

    def test_inverse_rank_var(self):
        """Verify the variance of the inverse rank."""
        var = np.reciprocal(self.ranks).var()
        numpy.testing.assert_allclose(var, harmonic_variances(n=self.n)[-1])


class WeightedTests(unittest.TestCase):
    """Tests for weighted aggregations."""

    def setUp(self) -> None:
        """Prepare input."""
        generator = np.random.default_rng()
        self.array = generator.random(size=(10,))

    def _test_equal_weights(self, func: Callable[[numpy.ndarray, Optional[numpy.ndarray]], numpy.ndarray]):
        """Verify that equal weights lead to unweighted results."""
        weights = np.full_like(self.array, fill_value=2.0)
        self.assertAlmostEqual(func(self.array, None).item(), func(self.array, weights).item())

    def test_weighted_harmonic_mean(self):
        """Test weighted harmonic mean."""
        self._test_equal_weights(weighted_harmonic_mean)

    def test_weighted_median(self):
        """Test weighted median."""
        self._test_equal_weights(weighted_median)

    def _test_weighted_mean_moment(
        self,
        closed_form: Callable[[numpy.ndarray, Optional[numpy.ndarray]], numpy.ndarray],
        statistic: Callable[[numpy.ndarray], numpy.ndarray],
        key: str,
    ):
        """Check the analytic expectation / variance of weighted mean against bootstrapped confidence intervals."""
        generator = numpy.random.default_rng(seed=0)
        individual = generator.random(size=(13,))
        # x_i ~ N(mu_i, 1)
        value = individual if key == "loc" else numpy.sqrt(individual)
        samples = generator.normal(size=(1_000,) + individual.shape, **{key: value})

        for weights in (None, generator.random(size=individual.shape)):
            # closed-form solution
            closed = closed_form(individual, weights)
            # sampled confidence interval
            result = numpy.average(samples, weights=weights, axis=-1)
            low, high = bootstrap((result,), statistic=statistic).confidence_interval
            # check that closed-form is in confidence interval of sampled
            self.assertLessEqual(low, closed)
            self.assertLessEqual(closed, high)

    def test_weighted_mean_expectation(self):
        """Test weighted mean expectation."""
        self._test_weighted_mean_moment(closed_form=weighted_mean_expectation, statistic=numpy.mean, key="loc")

    def test_weighted_mean_variance(self):
        """Test weighted mean variance."""
        self._test_weighted_mean_moment(closed_form=weighted_mean_variance, statistic=numpy.var, key="scale")


def test_stable_product():
    """Test stable_product."""
    generator = numpy.random.default_rng(seed=0)
    array = generator.random(size=(13,))

    # positive values only
    numpy.testing.assert_almost_equal(stable_product(array), numpy.prod(array))
    numpy.testing.assert_almost_equal(stable_product(np.log(array), is_log=True), numpy.prod(array))

    # positive and negative values
    array = 2 * array - 1
    numpy.testing.assert_almost_equal(stable_product(array), numpy.prod(array))
