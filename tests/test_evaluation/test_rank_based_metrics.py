"""Tests for rank-based metrics."""

import unittest
from collections.abc import Callable, Sequence

import numpy
import numpy as np
import pytest
import unittest_templates
from scipy.stats import bootstrap

import pykeen.metrics.ranking
from pykeen.metrics.ranking import generalized_harmonic_numbers, harmonic_variances
from pykeen.metrics.utils import (
    compute_log_expected_power,
    compute_median_survival_function,
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


class ZInverseHarmonicMeanRankTests(cases.ZRankBasedMetricTestCase):
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


class ZArithmeticMeanRankTests(cases.ZRankBasedMetricTestCase):
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

    def test_weights_coherence(self) -> None:
        # TODO: do we want this interpretation?
        raise unittest.SkipTest("The weights of a geometric mean do not represent sample weights.")


class ZGeometricMeanRankTests(cases.ZRankBasedMetricTestCase):
    """Tests for z-geometric mean rank."""

    cls = pykeen.metrics.ranking.ZGeometricMeanRank


class HarmonicMeanRankTests(cases.RankBasedMetricTestCase):
    """Tests for harmonic mean rank."""

    cls = pykeen.metrics.ranking.HarmonicMeanRank


class HitsAtKTests(cases.RankBasedMetricTestCase):
    """Tests for Hits at k."""

    cls = pykeen.metrics.ranking.HitsAtK


class ZHitsAtKTests(cases.ZRankBasedMetricTestCase):
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

    def test_weights_direction(self) -> None:
        raise unittest.SkipTest("Test does not make sense to dispersion metrics")


class MedianRankTests(cases.RankBasedMetricTestCase):
    """Tests for median rank."""

    cls = pykeen.metrics.ranking.MedianRank


class StandardDeviationTests(cases.RankBasedMetricTestCase):
    """Tests for rank standard deviation."""

    cls = pykeen.metrics.ranking.StandardDeviation

    def test_weights_direction(self) -> None:
        raise unittest.SkipTest("Test does not make sense to dispersion metrics")


class VarianceTests(cases.RankBasedMetricTestCase):
    """Tests for rank variance."""

    cls = pykeen.metrics.ranking.Variance

    def test_weights_direction(self) -> None:
        raise unittest.SkipTest("Test does not make sense to dispersion metrics")


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

    def _test_equal_weights(self, func: Callable[[numpy.ndarray, numpy.ndarray | None], numpy.ndarray]):
        """Verify that equal weights lead to unweighted results."""
        weights = np.full_like(self.array, fill_value=2.0)
        assert func(self.array, None).item() == pytest.approx(func(self.array, weights).item())

    def test_weighted_harmonic_mean(self):
        """Test weighted harmonic mean."""
        self._test_equal_weights(weighted_harmonic_mean)

    def test_weighted_median(self):
        """Test weighted median."""
        self._test_equal_weights(weighted_median)

    def _test_weighted_mean_moment(
        self,
        closed_form: Callable[[numpy.ndarray, numpy.ndarray | None], numpy.ndarray],
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
            assert low <= closed
            assert closed <= high

    def test_weighted_mean_expectation(self):
        """Test weighted mean expectation."""
        self._test_weighted_mean_moment(closed_form=weighted_mean_expectation, statistic=numpy.mean, key="loc")

    def test_weighted_mean_variance(self):
        """Test weighted mean variance."""
        self._test_weighted_mean_moment(closed_form=weighted_mean_variance, statistic=numpy.var, key="scale")


def _compute_log_expected_power_reference_single(k: int, p: float) -> float:
    """Compute the reference value for compute_log_expected_power."""
    # ln( E[X^p] )
    ks = np.arange(1, k + 1, dtype=np.float64)
    k_sum = (ks**p).sum()
    return np.log(k_sum) - np.log(k)


def _compute_test_compute_log_expected_power_reference(ks: Sequence[int], ps: Sequence[float]) -> float:
    """Compute the reference value for compute_log_expected_power."""
    # $sum( ln( E[X_i^p_i] ) )$.
    return sum(_compute_log_expected_power_reference_single(k=k, p=p) for k, p in zip(ks, ps, strict=True))


@pytest.mark.parametrize(
    ("ks", "ps"),
    [
        # Single elements with various k and p values
        ([5], [-1.0]),
        ([5], [0.0]),
        ([5], [1.0]),
        ([10], [-1.0]),
        ([10], [2.0]),
        ([50], [0.5]),
        ([100], [3.0]),
        # Multiple elements
        ([3, 5, 7], [1.0, 2.0, 0.5]),
        ([10, 20, 30], [0.0, 1.0, 2.0]),
        ([5, 10, 15, 20], [1.0, 1.5, 2.0, 0.5]),
    ],
)
def test_compute_log_expected_power(ks: list[int], ps: list[float]) -> None:
    """Test compute_log_expected_power with single and multiple elements."""
    k_values = np.array(ks)
    powers = np.array(ps)

    expected = _compute_test_compute_log_expected_power_reference(ks, ps)
    result = compute_log_expected_power(k_values, powers)
    assert result == pytest.approx(expected, rel=1e-10)


def test_compute_log_expected_power_batching_consistency() -> None:
    """Test that different memory limits produce the same result."""
    # Use a moderate-sized input
    rng = np.random.default_rng(42)
    k_values = rng.integers(10, 100, size=50)
    powers = rng.uniform(0.5, 3.0, size=50)

    # Compute reference value
    expected = _compute_test_compute_log_expected_power_reference(k_values.tolist(), powers.tolist())

    # Compute with different memory limits - all should match the reference
    result_large = compute_log_expected_power(k_values, powers, memory_limit_elements=10**7)
    result_small = compute_log_expected_power(k_values, powers, memory_limit_elements=100)
    result_tiny = compute_log_expected_power(k_values, powers, memory_limit_elements=10)

    assert result_large == pytest.approx(expected, rel=1e-10)
    assert result_small == pytest.approx(expected, rel=1e-10)
    assert result_tiny == pytest.approx(expected, rel=1e-10)


def test_compute_log_expected_power_sorted_vs_unsorted():
    """Test that the function handles both sorted and unsorted k_values correctly."""
    # Base data
    k_values = np.array([5, 10, 15, 20])
    powers = np.array([1.0, 1.5, 2.0, 0.5])

    # Indices for different orderings
    unsorted_idx = np.array([2, 0, 3, 1])  # reorders to [15, 5, 20, 10]

    result_sorted = compute_log_expected_power(k_values, powers)
    result_unsorted = compute_log_expected_power(k_values[unsorted_idx], powers[unsorted_idx])

    # Should get the same result regardless of input order
    assert result_sorted == pytest.approx(result_unsorted, rel=1e-10)


def _assert_valid_survival_function(sf: np.ndarray, k_max: int, atol: float = 0.0) -> None:
    """Assert that the given array is a valid survival function.

    :param sf: The survival function array
    :param k_max: The maximum $k$.
    :param atol: Absolute tolerance for boundary checks and monotonicity
    """
    # 0 to k_max
    assert len(sf) == k_max + 1
    # All probabilities should be in [0, 1]
    assert all(sf >= 0.0)
    assert all(sf <= 1.0)
    # Starts at 1
    assert sf[0] == pytest.approx(1.0, abs=atol)
    # Ends at 0
    assert sf[-1] == pytest.approx(0.0, abs=atol)
    # Monotonically non-increasing
    # Check monotonicity: sf[i] >= sf[i+1]
    differences = np.diff(sf)
    assert all(differences <= atol)


def test_median_survival_function_single_candidate():
    """Test with a single variable."""
    # Single variable with k=10
    num_candidates = np.array([10])
    sf = compute_median_survival_function(num_candidates)

    # For single variable, median is just that variable
    # P(X > x) = (k - x) / k for x in [0, k]
    expected = np.linspace(1.0, 0.0, 11)
    numpy.testing.assert_allclose(sf, expected, atol=1e-10)


@pytest.mark.parametrize(
    "num_candidates",
    [
        pytest.param(np.full(3, 5), id="uniform_candidates"),
        pytest.param(np.array([3, 5, 7]), id="different_candidates"),
        pytest.param(np.full(2, 4), id="two_variables_even"),
        pytest.param(np.full(3, 6), id="three_variables_odd"),
    ],
)
def test_median_survival_function_basic_properties(num_candidates: np.ndarray):
    """Test basic survival function properties for various candidate configurations."""
    sf = compute_median_survival_function(num_candidates)
    _assert_valid_survival_function(sf, k_max=int(num_candidates.max()))


def test_median_survival_function_against_simulation():
    """Test against empirical simulation for validation."""
    generator = numpy.random.default_rng(seed=42)
    num_candidates = np.array([10, 15, 20])

    # Compute analytical survival function
    sf = compute_median_survival_function(num_candidates)

    # Run simulation
    n_samples = 10_000
    samples = np.array([np.median([generator.integers(1, k + 1) for k in num_candidates]) for _ in range(n_samples)])

    # Compute empirical survival function
    empirical_sf = np.array([(samples > x).mean() for x in range(len(sf))])

    # The analytical result should be close to empirical (with some tolerance)
    numpy.testing.assert_allclose(sf, empirical_sf, rtol=0.05)
