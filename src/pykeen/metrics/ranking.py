# -*- coding: utf-8 -*-

"""Ranking metrics."""

import math
from abc import abstractmethod
from typing import ClassVar, Collection, Iterable, Optional, Tuple, Type

import numpy as np
from class_resolver import ClassResolver
from docdata import parse_docdata
from scipy import stats

from .utils import Metric, ValueRange
from ..typing import RANK_REALISTIC, RANK_TYPES, RankType

__all__ = [
    "rank_based_metric_resolver",
    # Base classes
    "RankBasedMetric",
    "BaseZMixin",
    "IncreasingZMixin",
    "DecreasingZMixin",
    "ExpectationNormalizedMixin",
    "ReindexMixin",
    # Concrete classes
    "ArithmeticMeanRank",
    "AdjustedArithmeticMeanRank",
    "AdjustedArithmeticMeanRankIndex",
    "ZArithmeticMeanRank",
    "InverseArithmeticMeanRank",
    #
    "GeometricMeanRank",
    "InverseGeometricMeanRank",
    #
    "HarmonicMeanRank",
    "InverseHarmonicMeanRank",
    "AdjustedInverseHarmonicMeanRank",
    "ZInverseHarmonicMeanRank",
    #
    "MedianRank",
    "InverseMedianRank",
    #
    "HitsAtK",
    "AdjustedHitsAtK",
    "ZHitsAtK",
    #
    "StandardDeviation",
    "Variance",
    "Count",
    #
    "HITS_METRICS",
]
EPSILON = 1.0e-12


class RankBasedMetric(Metric):
    """A base class for rank-based metrics."""

    # rank based metrics do not need binarized scores
    binarize: ClassVar[bool] = False

    #: the supported rank types. Most of the time equal to all rank types
    supported_rank_types: ClassVar[Collection[RankType]] = RANK_TYPES

    #: whether the metric requires the number of candidates for each ranking task
    needs_candidates: ClassVar[bool] = False

    #: whether the metric supports weights
    supports_weights: ClassVar[bool] = False

    @abstractmethod
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate the metric.

        :param ranks: shape: s
            the individual ranks
        :param num_candidates: shape: s
            the number of candidates for each individual ranking task
        :param weights: shape: s
            the weights for the individual ranks
        """
        raise NotImplementedError

    def _yield_sampled_values(
        self,
        num_candidates: np.ndarray,
        num_samples: int,
        generator: Optional[np.random.Generator] = None,
    ) -> Iterable[float]:
        num_candidates = np.asarray(num_candidates)
        if generator is None:
            generator = np.random.default_rng()
        for _ in range(num_samples):
            yield self(generator.integers(low=1, high=num_candidates + 1))

    def numeric_expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: int,
        generator: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Compute expected metric value by summation.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation
        :param generator:
            A random number generator
        :return:
            The estimated expected value of this metric

        .. warning ::

            Depending on the metric, the estimate may not be very accurate and converge slowly, cf.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
        """
        return (
            sum(self._yield_sampled_values(num_candidates=num_candidates, num_samples=num_samples, generator=generator))
            / num_samples
        )

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        """Compute expected metric value.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation, if no closed form
            expected value is implemented
        :return:
            The expected value of this metric
        :raises ValueError:
            Raised if a closed form variance has not been implemented and no
            number of samples are given

        .. note::

            Prefers analytical solution, if available, but falls back to numeric
            estimation via summation, cf. :func:`numeric_expected_value`.
        """
        if num_samples is None:
            raise ValueError("Numeric estimation requires to specify a number of samples.")
        return self.numeric_expected_value(num_candidates=num_candidates, num_samples=num_samples)

    def numeric_variance(
        self, num_candidates: np.ndarray, num_samples: int, generator: Optional[np.random.Generator] = None
    ) -> float:
        """Compute variance by summation.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation
        :param generator:
            A random number generator
        :return:
            The estimated variance of this metric
        """
        num_candidates = np.asarray(num_candidates)
        if generator is None:
            generator = np.random.default_rng()
        ranks = generator.integers(low=1, high=num_candidates + 1, size=(num_samples, *num_candidates.shape))
        return np.apply_along_axis(self, 0, ranks).var().item()

    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        """Compute variance.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation, if no closed form
            expected value is implemented
        :return:
            The variance of this metric
        :raises ValueError:
            Raised if a closed form variance has not been implemented and no
            number of samples are given

        .. note::
            Prefers analytical solution, if available, but falls back to numeric
            estimation via summation, cf. :func:`numeric_variance`.
        """
        if num_samples is None:
            raise ValueError("Numeric estimation requires to specify a number of samples.")
        return self.numeric_variance(num_candidates=num_candidates, num_samples=num_samples)

    def std(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        """Compute the standard deviation."""
        return math.sqrt(self.variance(num_candidates=num_candidates, num_samples=num_samples))


class BaseZMixin(RankBasedMetric):
    """A base class for creating a z-scored metric."""

    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True
    value_range = ValueRange(lower=None, upper=None)

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:  # noqa: D102
        return 0.0  # centered

    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:  # noqa: D102
        return 1.0  # re-scaled


def _safe_divide(x: float, y: float) -> float:
    return x / max(y, EPSILON)


class IncreasingZMixin(BaseZMixin):
    """A mixin to create a z-scored metric.

    .. warning:: This requires a closed-form solution to the expected value and variance
    """

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        v = super().expected_value(num_candidates=num_candidates) - super().__call__(
            ranks=ranks, num_candidates=num_candidates
        )
        std = super().std(num_candidates=num_candidates)
        return _safe_divide(v, std)


class DecreasingZMixin(BaseZMixin):
    """A mixin to create a z-scored metric.

    .. warning:: This requires a closed-form solution to the expected value and variance
    """

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        v = super().__call__(ranks=ranks, num_candidates=num_candidates) - super().expected_value(
            num_candidates=num_candidates
        )
        std = super().std(num_candidates=num_candidates)
        return _safe_divide(v, std)


class ExpectationNormalizedMixin(RankBasedMetric):
    """A mixin to create an expectation-normalized metric.

    .. warning:: This requires a closed-form solution to the expected value
    """

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return super().__call__(ranks=ranks, num_candidates=num_candidates) / super().expected_value(
            num_candidates=num_candidates
        )

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:  # noqa: D102
        return 1.0  # centered


class ReindexMixin(RankBasedMetric):
    r"""A mixin to create an expectation normalized metric with max of 1 and expectation of 0.

    .. math::

        \mathbb{M}^{*} = $\frac{\mathbb{M} - \mathbb{E}[\mathbb{M}]}{1 - \mathbb{E}[\mathbb{M}]}

    .. warning:: This requires a closed-form solution to the expected value
    """

    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        ev = super().expected_value(num_candidates=num_candidates)
        return _safe_divide(super().__call__(ranks=ranks, num_candidates=num_candidates) - ev, 1 - ev)


@parse_docdata
class ArithmeticMeanRank(RankBasedMetric):
    """The (arithmetic) mean rank.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#mean-rank
    description: The arithmetic mean over all ranks.
    """

    name = "Mean Rank (MR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("mean_rank", "mr")
    supports_weights: ClassVar[bool] = True

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        if weights is None:
            return np.asanyarray(ranks).mean().item()
        return _safe_divide((ranks * weights).sum().item(), weights.sum().item())

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        r"""
        Calculate the expected mean rank under random ordering.

        .. math ::

            E[MR] = \frac{1}{n} \sum \limits_{i=1}^{n} \frac{1 + CSS[i]}{2}
                = \frac{1}{2}(1 + \frac{1}{n} \sum \limits_{i=1}^{n} CSS[i])

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation

        :return:
            the expected value of the mean rank
        """
        return 0.5 * (1 + np.asanyarray(num_candidates).mean().item())

    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        """Calculate the variance under random ordering.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation

        :return:
            the variance of the mean rank
        """
        n = np.asanyarray(num_candidates).mean().item()
        return (n ** 2 - 1) / 12.0


@parse_docdata
class ZArithmeticMeanRank(IncreasingZMixin, ArithmeticMeanRank):
    """The z-scored arithmetic mean rank.

    ---
    link: https://github.com/pykeen/pykeen/pull/814
    description: The z-scored mean rank
    """

    name = "z-Mean Rank (ZMR)"
    synonyms: ClassVar[Collection[str]] = ("zamr", "zmr")


@parse_docdata
class InverseArithmeticMeanRank(RankBasedMetric):
    """The inverse arithmetic mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The inverse of the arithmetic mean over all ranks.
    """

    name = "Inverse Arithmetic Mean Rank (IAMR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True
    synonyms: ClassVar[Collection[str]] = ("iamr",)

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(np.asanyarray(ranks).mean()).item()


@parse_docdata
class GeometricMeanRank(RankBasedMetric):
    """The geometric mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The geometric mean over all ranks.
    """

    name = "Geometric Mean Rank (GMR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("gmr",)

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return stats.gmean(ranks).item()


@parse_docdata
class InverseGeometricMeanRank(RankBasedMetric):
    """The inverse geometric mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The inverse of the geometric mean over all ranks.
    """

    name = "Inverse Geometric Mean Rank (IGMR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True
    synonyms: ClassVar[Collection[str]] = ("igmr",)

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(stats.gmean(ranks)).item()


@parse_docdata
class HarmonicMeanRank(RankBasedMetric):
    """The harmonic mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The harmonic mean over all ranks.
    """

    name = "Harmonic Mean Rank (HMR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("hmr",)

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return stats.hmean(ranks).item()


@parse_docdata
class InverseHarmonicMeanRank(RankBasedMetric):
    """The inverse harmonic mean rank.

    ---
    link: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    description: The inverse of the harmonic mean over all ranks.
    """

    name = "Mean Reciprocal Rank (MRR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("mean_reciprocal_rank", "mrr")
    increasing = True

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(stats.hmean(ranks)).item()

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        r"""
        Calculate the expected mean rank under random ordering.

        .. math ::

            \mathbb{E}\left[\textrm{MRR}\right]
            = \mathbb{E}\left[\frac{1}{n} \sum \limits_{i=1}^n r_i^{-1}\right]
            = \frac{1}{n} \sum \limits_{i=1}^n \mathbb{E}\left[r_i^{-1}\right]
            = \mathbb{E}\left[r_i^{-1}\right]
            \stackrel{*}{=} \frac{\ln n}{n - 1}

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation

        :return:
            the expected mean rank
        """
        n = np.asanyarray(num_candidates).mean().item()
        return _safe_divide(math.log(n), n - 1)

    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:  # noqa:D102
        n = np.asanyarray(num_candidates).mean()
        return (1 / n - (np.log(n) / (n - 1)) ** 2).item()


@parse_docdata
class AdjustedInverseHarmonicMeanRank(ReindexMixin, InverseHarmonicMeanRank):
    r"""The adjusted MRR index.

    .. note ::
        the actual lower bound is $\frac{-\mathbb{E}[\text{MRR}]}{1-\mathbb{E}[\text{MRR}]}$,
        and thus data dependent.

    ---
    link: https://github.com/pykeen/pykeen/pull/814
    description: The re-indexed adjusted MRR
    tight_lower: -E[f]/(1-E[f])
    """

    name = "Adjusted Inverse Harmonic Mean Rank"
    synonyms: ClassVar[Collection[str]] = ("amrr", "aihmr", "adjusted_mrr", "adjusted_mean_reciprocal_rank")
    value_range = ValueRange(lower=None, lower_inclusive=False, upper=1, upper_inclusive=True)


@parse_docdata
class ZInverseHarmonicMeanRank(DecreasingZMixin, InverseHarmonicMeanRank):
    """The z-inverse harmonic mean rank (ZIHMR).

    ---
    link: https://github.com/pykeen/pykeen/pull/814
    description: The z-scored mean reciprocal rank
    """

    name = "z-Mean Reciprocal Rank (ZMRR)"
    synonyms: ClassVar[Collection[str]] = ("zmrr", "zihmr")


@parse_docdata
class MedianRank(RankBasedMetric):
    """The median rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The median over all ranks.
    """

    name = "Median Rank"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.median(ranks).item()

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:  # noqa: D102
        # TODO: not proven (yet)
        m = num_candidates.max()
        # performance trick: O(num_triples * max_count) => O(num_unique_candidates * max_count)
        uniq, count = np.unique(num_candidates, return_counts=True)
        ps = np.zeros(shape=(m,))
        for c, w in zip(uniq, count):
            ps[:c] += w * np.full(shape=(c,), fill_value=1 / c)
        # normalize cdf
        ps = np.cumsum(ps)
        ps /= ps[-1]
        # find median
        idx = np.searchsorted(ps, v=0.5, side="left")
        # special case
        if idx == 0:
            # zero- to one-based
            return idx + 1
        # linear interpolation
        p_upper = ps[idx]
        p_lower = ps[idx - 1]
        idx = idx - (0.5 - p_lower) / (p_upper - p_lower)
        # zero- to one-based
        return idx + 1


@parse_docdata
class InverseMedianRank(RankBasedMetric):
    """The inverse median rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The inverse of the median over all ranks.
    """

    name = "Inverse Median Rank"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(np.median(ranks)).item()


@parse_docdata
class StandardDeviation(RankBasedMetric):
    """The ranks' standard deviation.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    name = "Standard Deviation (std)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_std", "std")

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.asanyarray(ranks).std().item()


@parse_docdata
class Variance(RankBasedMetric):
    """The ranks' variance.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    name = "Variance"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_var", "var")

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.asanyarray(ranks).var().item()


@parse_docdata
class MedianAbsoluteDeviation(RankBasedMetric):
    """The ranks' median absolute deviation (MAD).

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    name = "Median Absolute Deviation (MAD)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_mad", "mad")

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return stats.median_abs_deviation(ranks, scale="normal").item()


@parse_docdata
class Count(RankBasedMetric):
    """The ranks' count.

    Lower numbers may indicate unreliable results.
    ---
    link: https://pykeen.readthedocs.io/en/stable/reference/evaluation.html
    """

    name = "Count"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_count",)

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return float(np.asanyarray(ranks).size)


@parse_docdata
class HitsAtK(RankBasedMetric):
    """The Hits @ k.

    ---
    description: The relative frequency of ranks not larger than a given k.
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#hits-k
    """

    name = "Hits @ K"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("h@k", "hits@k", "h@", "hits@", "hits_at_", "h_at_")
    increasing = True

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = k

    def _extra_repr(self) -> Iterable[str]:
        yield f"k={self.k}"

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.less_equal(ranks, self.k).mean().item()

    @property
    def key(self) -> str:  # noqa: D102
        return super().key[:-1] + str(self.k)

    def expected_value(self, num_candidates: np.ndarray, num_samples: Optional[int] = None) -> float:
        r"""
        Calculate the expected Hits@k under random ordering.

        .. math ::

            E[Hits@k] = \frac{1}{n} \sum \limits_{i=1}^{n} min(\frac{k}{CSS[i]}, 1.0)

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation

        :return:
            the expected Hits@k value
        """
        return (
            self.k * np.reciprocal(np.asanyarray(num_candidates, dtype=float)).clip(min=None, max=1.0 / self.k).mean()
        ).item()

    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:  # noqa:D102
        e = self.expected_value(num_candidates=num_candidates, num_samples=num_samples)
        return e * (1 - e)


@parse_docdata
class AdjustedHitsAtK(ReindexMixin, HitsAtK):
    r"""The adjusted Hits at K ($AH_k$).

    .. note ::
        the actual lower bound is $\frac{-\mathbb{E}[H_k]}{1 - \mathbb{E}[H_k]}$, and thus data dependent.

    ---
    link: https://github.com/pykeen/pykeen/pull/814
    description: The re-indexed adjusted hits at K
    tight_lower: -E[f]/(1-E[f])
    """

    name = "Adjusted Hits at K"
    synonyms: ClassVar[Collection[str]] = (
        "ahk",
        "ah@k",
        "ahits@k",
        "ah@",
        "ahits@",
        "ahits_at_",
        "ah_at_",
        "adjusted_hits_at_",
    )
    value_range = ValueRange(lower=None, lower_inclusive=False, upper=1, upper_inclusive=True)


@parse_docdata
class ZHitsAtK(DecreasingZMixin, HitsAtK):
    """The z-scored hits at k ($ZAH_k$).

    ---
    link: https://github.com/pykeen/pykeen/pull/814
    description: The z-scored hits at K
    """

    name = "z-Hits at K"
    synonyms: ClassVar[Collection[str]] = ("z_hits_at_", "zahk")
    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True


@parse_docdata
class AdjustedArithmeticMeanRank(ExpectationNormalizedMixin, ArithmeticMeanRank):
    """The adjusted arithmetic mean rank (AMR).

    ---
    description: The mean over all ranks divided by its expected value.
    link: https://arxiv.org/abs/2002.06914
    """

    name = "Adjusted Arithmetic Mean Rank (AAMR)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=2, upper_inclusive=False)
    synonyms: ClassVar[Collection[str]] = ("adjusted_mean_rank", "amr", "aamr")
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True
    increasing = False


@parse_docdata
class AdjustedArithmeticMeanRankIndex(ArithmeticMeanRank):
    """The adjusted arithmetic mean rank index (AMRI).

    ---
    link: https://arxiv.org/abs/2002.06914
    description: The re-indexed adjusted mean rank (AAMR)
    """

    name = "Adjusted Arithmetic Mean Rank Index (AAMRI)"
    value_range = ValueRange(lower=-1, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("adjusted_mean_rank_index", "amri", "aamri")
    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True

    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return 1.0 - _safe_divide(
            super().__call__(ranks=ranks, num_candidates=num_candidates) - 1.0,
            # E[MR-1] is equivalent to E[MR]-1
            super().expected_value(num_candidates=num_candidates) - 1.0,
        )

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        """Analytically calculate the expected value."""
        return 0.0


rank_based_metric_resolver: ClassResolver[RankBasedMetric] = ClassResolver.from_subclasses(
    base=RankBasedMetric,
    default=InverseHarmonicMeanRank,  # mrr
    skip={BaseZMixin, IncreasingZMixin, DecreasingZMixin, ExpectationNormalizedMixin, ReindexMixin},
)

HITS_METRICS: Tuple[Type[RankBasedMetric], ...] = (HitsAtK, ZHitsAtK, AdjustedHitsAtK)
