"""(Rank-Based) Metrics."""

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Collection, Optional

import numpy as np
from class_resolver import Resolver
from scipy import stats

from .expectation import expected_mean_rank
from ..typing import RANK_REALISTIC, RANK_TYPES, RankType

__all__ = [
    "metric_resolver",
]


@dataclass
class ValueRange:
    """A value range description."""

    #: the lower bound
    lower: Optional[float] = None

    #: whether the lower bound is inclusive
    lower_inclusive: bool = False

    #: the upper bound
    upper: Optional[float] = None

    #: whether the upper bound is inclusive
    upper_inclusive: bool = False

    def __contains__(self, x: float) -> bool:
        """Test whether a value is contained in the value range."""
        if self.lower is not None:
            if x < self.lower:
                return False
            if not self.lower_inclusive and x == self.lower:
                return False
        if self.upper is not None:
            if x > self.upper:
                return False
            if not self.upper_inclusive and x == self.upper:
                return False
        return True


class RankBasedMetric:
    """A base class for rank-based metrics."""

    #: whether it is increasing, i.e., larger values are better
    increasing: ClassVar[bool] = False

    #: the value range (as string)
    value_range: ClassVar[Optional[ValueRange]] = None

    #: the supported rank types. Most of the time equal to all rank types
    supported_rank_types: ClassVar[Collection[RankType]] = RANK_TYPES

    #: synonyms for this metric
    synonyms: ClassVar[Collection[str]] = tuple()

    #: whether the metric requires the number of candidates for each ranking task
    needs_candidates: ClassVar[bool] = False

    @abstractmethod
    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:
        """
        Evaluate the metric.

        :param ranks: shape: s
            the individual ranks
        :param num_candidates: shape: s
            the number of candidates for each individual ranking task
        """
        raise NotImplementedError


class ArithmeticMeanRank(RankBasedMetric):
    """The (arithmetic) mean rank."""

    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    synonyms = ("mean_rank", "mr")

    @staticmethod
    def call(ranks: np.ndarray) -> float:
        """Evaluate the arithmetic mean rank."""
        return np.mean(ranks).item()

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return ArithmeticMeanRank.call(ranks)


class InverseArithmeticMeanRank(RankBasedMetric):
    """The inverse arithmetic mean rank."""

    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(np.mean(ranks)).item()


class GeometricMeanRank(RankBasedMetric):
    """The geometric mean rank."""

    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    synonyms = ("gmr",)

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return stats.gmean(ranks).item()


class InverseGeometricMeanRank(RankBasedMetric):
    """The inverse geometric mean rank."""

    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(stats.gmean(ranks)).item()


class HarmonicMeanRank(RankBasedMetric):
    """The harmonic mean rank."""

    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    synonyms = ("hmr",)

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return stats.hmean(ranks).item()


class InverseHarmonicMeanRank(RankBasedMetric):
    """The inverse harmonic mean rank."""

    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    synonyms = ("mean_reciprocal_rank", "mrr")
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(stats.hmean(ranks)).item()


class MedianRank(RankBasedMetric):
    """The median rank."""

    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.median(ranks).item()


class InverseMedianRank(RankBasedMetric):
    """The inverse median rank."""

    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(np.median(ranks)).item()


class StandardDeviation(RankBasedMetric):
    """The ranks' standard deviation."""

    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    synonyms = ("rank_std", "std")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.std(ranks).item()


class Variance(RankBasedMetric):
    """The ranks' variance."""

    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    synonyms = ("rank_var", "var")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.var(ranks).item()


class MedianAbsoluteDeviation(RankBasedMetric):
    """The ranks' median absolute deviation (MAD)."""

    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    synonyms = ("rank_mad", "mad")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return stats.median_abs_deviation(ranks, scale="normal").item()


class Count(RankBasedMetric):
    """The ranks' count."""

    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return float(ranks.size)


class HitsAtK(RankBasedMetric):
    """The Hits@k."""

    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms = ("h@k", "hits@k", "h@", "hits@", "hits_at_", "h_at_")
    increasing = True

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = k

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.less_equal(ranks, self.k).mean().item()


class AdjustedArithmeticMeanRank(RankBasedMetric):
    """The adjusted arithmetic mean rank (AMR)."""

    value_range = ValueRange(lower=0, lower_inclusive=True, upper=2, upper_inclusive=False)
    synonyms = ("adjusted_mean_rank", "amr", "aamr")
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return ArithmeticMeanRank.call(ranks) / expected_mean_rank(num_candidates=num_candidates)


class AdjustedArithmeticMeanRankIndex(RankBasedMetric):
    """The adjusted arithmetic mean rank index (AMRI)."""

    value_range = ValueRange(lower=-1, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms = ("adjusted_mean_rank_index", "amri", "aamri")
    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return 1.0 - (
            (ArithmeticMeanRank.call(ranks) - 1.0) / (expected_mean_rank(num_candidates=num_candidates) - 1.0)
        )


metric_resolver: Resolver[RankBasedMetric] = Resolver.from_subclasses(
    base=RankBasedMetric,
    default=InverseArithmeticMeanRank,  # mrr
)
