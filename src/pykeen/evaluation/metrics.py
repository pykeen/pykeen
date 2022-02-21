# -*- coding: utf-8 -*-

"""Utilities for metrics."""

import itertools as itt
import math
import re
from abc import abstractmethod
from typing import ClassVar, Collection, Iterable, Mapping, NamedTuple, Optional, Union, cast

import numpy as np
from class_resolver import Resolver
from docdata import get_docdata, parse_docdata
from scipy import stats

from .utils import ValueRange
from ..typing import (
    RANK_REALISTIC,
    RANK_TYPE_SYNONYMS,
    RANK_TYPES,
    SIDE_BOTH,
    SIDES,
    ExtendedRankType,
    ExtendedTarget,
    RankType,
)

__all__ = [
    "MetricKey",
    "rank_based_metric_resolver",
]

# TODO: get rid of constants
ARITHMETIC_MEAN_RANK = "arithmetic_mean_rank"  # also known as mean rank (MR)
GEOMETRIC_MEAN_RANK = "geometric_mean_rank"
HARMONIC_MEAN_RANK = "harmonic_mean_rank"
MEDIAN_RANK = "median_rank"
INVERSE_ARITHMETIC_MEAN_RANK = "inverse_arithmetic_mean_rank"
INVERSE_GEOMETRIC_MEAN_RANK = "inverse_geometric_mean_rank"
INVERSE_HARMONIC_MEAN_RANK = "inverse_harmonic_mean_rank"  # also known as mean reciprocal rank (MRR)
INVERSE_MEDIAN_RANK = "inverse_median_rank"
ADJUSTED_ARITHMETIC_MEAN_RANK = "adjusted_arithmetic_mean_rank"
ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX = "adjusted_arithmetic_mean_rank_index"
RANK_STD = "rank_std"
RANK_VARIANCE = "rank_var"
RANK_MAD = "rank_mad"
RANK_COUNT = "rank_count"
TYPES_REALISTIC_ONLY = {ADJUSTED_ARITHMETIC_MEAN_RANK, ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX}
METRIC_SYNONYMS = {
    "adjusted_mean_rank": ADJUSTED_ARITHMETIC_MEAN_RANK,
    "adjusted_mean_rank_index": ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX,
    "amr": ADJUSTED_ARITHMETIC_MEAN_RANK,
    "aamr": ADJUSTED_ARITHMETIC_MEAN_RANK,
    "amri": ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX,
    "aamri": ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX,
    "igmr": INVERSE_GEOMETRIC_MEAN_RANK,
    "iamr": INVERSE_ARITHMETIC_MEAN_RANK,
    "mr": ARITHMETIC_MEAN_RANK,
    "mean_rank": ARITHMETIC_MEAN_RANK,
    "mrr": INVERSE_HARMONIC_MEAN_RANK,
    "mean_reciprocal_rank": INVERSE_HARMONIC_MEAN_RANK,
}

_SIDE_PATTERN = "|".join(SIDES)
_TYPE_PATTERN = "|".join(itt.chain(RANK_TYPES, RANK_TYPE_SYNONYMS.keys()))
METRIC_PATTERN = re.compile(
    rf"(\.(?P<side>{_SIDE_PATTERN}))?(\.(?P<type>{_TYPE_PATTERN}))?(?P<name>[\w@]+)(\.(?P<k>\d+))?",
)
HITS_PATTERN = re.compile(r"(hits_at_|hits@|h@)(?P<k>\d+)")


class MetricKey(NamedTuple):
    """A key for the kind of metric to resolve."""

    #: Name of the metric
    name: str
    #: Side of the metric, or "both"
    side: ExtendedTarget
    #: The rank type
    rank_type: ExtendedRankType
    #: The k if this represents a hits at k metric
    k: Optional[int]

    def __str__(self) -> str:  # noqa: D105
        name = self.name
        if self.k:
            name = name[:-1] + str(self.k)
        return ".".join((self.side, self.rank_type, name))

    @classmethod
    def lookup(cls, s: str) -> "MetricKey":
        """Functional metric name normalization."""
        match = METRIC_PATTERN.match(s)
        if not match:
            raise ValueError(f"Invalid metric name: {s}")
        k: Union[None, str, int]
        name, side, rank_type, k = [match.group(key) for key in ("name", "side", "type", "k")]

        # normalize metric name
        if not name:
            raise ValueError("A metric name must be provided.")
        # handle spaces and case
        name = name.lower().replace(" ", "_")

        # special case for hits_at_k
        match = HITS_PATTERN.match(name)
        if match:
            name = "hits_at_k"
            k = match.group("k")
        if name == "hits_at_k":
            if k is None:
                k = 10
            # TODO: Fractional?
            try:
                k = int(k)
            except ValueError as error:
                raise ValueError(f"Invalid k={k} for hits_at_k") from error
            if k < 0:
                raise ValueError(f"For hits_at_k, you must provide a positive value of k, but found {k}.")
        assert k is None or isinstance(k, int)

        # synonym normalization
        name = METRIC_SYNONYMS.get(name, name)

        # normalize side
        side = side or SIDE_BOTH
        side = side.lower()
        if side not in SIDES:
            raise ValueError(f"Invalid side: {side}. Allowed are {SIDES}.")

        # normalize rank type
        rank_type = rank_type or RANK_REALISTIC
        rank_type = rank_type.lower()
        rank_type = RANK_TYPE_SYNONYMS.get(rank_type, rank_type)
        if rank_type not in RANK_TYPES:
            raise ValueError(f"Invalid rank type: {rank_type}. Allowed are {RANK_TYPES}.")
        elif rank_type != RANK_REALISTIC and name in TYPES_REALISTIC_ONLY:
            raise ValueError(f"Invalid rank type for {name}: {rank_type}. Allowed type: {RANK_REALISTIC}")

        return cls(name, cast(ExtendedTarget, side), cast(ExtendedRankType, rank_type), k)

    @classmethod
    def normalize(cls, s: str) -> str:
        """Normalize a metric key string."""
        return str(cls.lookup(s))


camel_to_snake_pattern = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    """Convert camel-case to snake case."""
    # cf. https://stackoverflow.com/a/1176023
    return camel_to_snake_pattern.sub("_", name).lower()


class Metric:
    """A base class for metrics."""

    #: The key for use in metric result dictionaries
    key: ClassVar[str]

    #: The name of the metric
    name: ClassVar[str]

    #: a link to further information
    link: ClassVar[str]

    #: whether the metric needs binarized scores
    binarize: ClassVar[Optional[bool]] = None

    #: whether it is increasing, i.e., larger values are better
    increasing: ClassVar[bool]

    #: the value range (as string)
    value_range: ClassVar[Optional[ValueRange]] = None

    #: synonyms for this metric
    synonyms: ClassVar[Collection[str]] = tuple()

    @classmethod
    def get_description(cls) -> str:
        """Get the description."""
        docdata = get_docdata(cls)
        if docdata is not None and "description" in docdata:
            return docdata["description"]
        assert cls.__doc__ is not None
        return cls.__doc__.splitlines()[0]

    @classmethod
    def get_link(cls) -> str:
        """Get the link from the docdata."""
        docdata = get_docdata(cls)
        if docdata is None:
            raise TypeError
        return docdata["link"]

    def _extra_repr(self) -> Iterable[str]:
        return []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self._extra_repr())})"

    def compose_key(self) -> str:
        """Compose the metric key."""
        return self.key


class RankBasedMetric(Metric):
    """A base class for rank-based metrics."""

    # rank based metrics do not need binarized scores
    binarize: ClassVar[bool] = False

    #: the supported rank types. Most of the time equal to all rank types
    supported_rank_types: ClassVar[Collection[RankType]] = RANK_TYPES

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

    def numeric_expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: int,
    ) -> float:
        """
        Compute expected metric value by summation.

        Depending on the metric, the estimate may not be very accurate and converge slowly, cf.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
        """
        num_candidates = np.asarray(num_candidates)
        generator = np.random.default_rng()
        expectation = 0.0
        for _ in range(num_samples):
            ranks = generator.integers(low=0, high=num_candidates)
            expectation += self(ranks)
        return expectation / num_samples

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> float:
        """
        Compute expected metric value.

        Prefers analytical solution, if available, but falls back to numeric estimation via summation.
        Depending on the metric, the numeric estimate may not be very accurate and converge slowly, cf.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
        """
        if num_samples is None:
            raise ValueError("Numeric estimation requires to specify a number of samples.")
        return self.numeric_expected_value(num_candidates=num_candidates, num_samples=num_samples)

    def compose_extended_key(self, extended_target: ExtendedTarget, rank_type: RankType) -> Iterable[str]:
        """Compose the metric key."""
        yield self.compose_key()
        yield extended_target
        yield rank_type


@parse_docdata
class ArithmeticMeanRank(RankBasedMetric):
    """The (arithmetic) mean rank.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#mean-rank
    description: The arithmetic mean over all ranks.
    """

    key = ARITHMETIC_MEAN_RANK
    name = "Mean Rank (MR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("mean_rank", "mr")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.mean(ranks).item()

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

        :return:
            the expected mean rank
        """
        return 0.5 * (1 + np.mean(np.asanyarray(num_candidates)))


@parse_docdata
class InverseArithmeticMeanRank(RankBasedMetric):
    """The inverse arithmetic mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The inverse of the arithmetic mean over all ranks.
    """

    key = INVERSE_ARITHMETIC_MEAN_RANK
    name = "Inverse Arithmetic Mean Rank (IAMR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(np.mean(ranks)).item()


@parse_docdata
class GeometricMeanRank(RankBasedMetric):
    """The geometric mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The geometric mean over all ranks.
    """

    key = GEOMETRIC_MEAN_RANK
    name = "Geometric Mean Rank (GMR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms = ("gmr",)

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return stats.gmean(ranks).item()


@parse_docdata
class InverseGeometricMeanRank(RankBasedMetric):
    """The inverse geometric mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The inverse of the geometric mean over all ranks.
    """

    key = INVERSE_GEOMETRIC_MEAN_RANK
    name = "Inverse Geometric Mean Rank (IGMR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(stats.gmean(ranks)).item()


@parse_docdata
class HarmonicMeanRank(RankBasedMetric):
    """The harmonic mean rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The harmonic mean over all ranks.
    """

    key = HARMONIC_MEAN_RANK
    name = "Harmonic Mean Rank (HMR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms = ("hmr",)

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return stats.hmean(ranks).item()


@parse_docdata
class InverseHarmonicMeanRank(RankBasedMetric):
    """The inverse harmonic mean rank.

    ---
    link: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    description: The inverse of the harmonic mean over all ranks.
    """

    key = INVERSE_HARMONIC_MEAN_RANK
    name = "Mean Reciprocal Rank (MRR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    synonyms = ("mean_reciprocal_rank", "mrr")
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(stats.hmean(ranks)).item()


@parse_docdata
class MedianRank(RankBasedMetric):
    """The median rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The median over all ranks.
    """

    key = MEDIAN_RANK
    name = "Median Rank"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.median(ranks).item()


@parse_docdata
class InverseMedianRank(RankBasedMetric):
    """The inverse median rank.

    ---
    link: https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html
    description: The inverse of the median over all ranks.
    """

    key = INVERSE_MEDIAN_RANK
    name = "Inverse Median Rank"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.reciprocal(np.median(ranks)).item()


@parse_docdata
class StandardDeviation(RankBasedMetric):
    """The ranks' standard deviation.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    key = RANK_STD
    name = "Standard Deviation (std)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms = ("rank_std", "std")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.std(ranks).item()


@parse_docdata
class Variance(RankBasedMetric):
    """The ranks' variance.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    key = RANK_VARIANCE
    name = "Variance"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms = ("rank_var", "var")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.var(ranks).item()


@parse_docdata
class MedianAbsoluteDeviation(RankBasedMetric):
    """The ranks' median absolute deviation (MAD).

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    key = RANK_MAD
    name = "Median Absolute Deviation (MAD)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms = ("rank_mad", "mad")

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return stats.median_abs_deviation(ranks, scale="normal").item()


@parse_docdata
class Count(RankBasedMetric):
    """The ranks' count.

    Lower numbers may indicate unreliable results.
    ---
    link: https://pykeen.readthedocs.io/en/stable/reference/evaluation.html
    """

    key = RANK_COUNT
    name = "Count"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return float(ranks.size)


@parse_docdata
class HitsAtK(RankBasedMetric):
    """The Hits @ k.

    ---
    description: The relative frequency of ranks not larger than a given k.
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#hits-k
    """

    key = "hits_at_k"
    name = "Hits @ K"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms = ("h@k", "hits@k", "h@", "hits@", "hits_at_", "h_at_")
    increasing = True

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = k

    def _extra_repr(self) -> Iterable[str]:
        yield f"k={self.k}"

    def compose_extended_key(self, extended_target: ExtendedTarget, rank_type: RankType) -> Iterable[str]:
        yield from super().compose_extended_key(extended_target, rank_type)
        yield str(self.k)

    def compose_key(self) -> str:
        """Compose the metric key."""
        return self.key[:-1] + str(self.k)

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return np.less_equal(ranks, self.k).mean().item()

    def expected_value(self, num_candidates: np.ndarray, num_samples: Optional[int] = None) -> float:
        r"""
        Calculate the expected Hits@k under random ordering.

        .. math ::

            E[Hits@k] = \frac{1}{n} \sum \limits_{i=1}^{n} min(\frac{k}{CSS[i]}, 1.0)

        :param num_candidates:
            the number of candidates for each individual rank computation

        :return:
            the expected Hits@k value
        """
        return self.k * np.mean(
            np.reciprocal(np.asanyarray(num_candidates, dtype=float)).clip(min=None, max=1 / self.k)
        )


@parse_docdata
class AdjustedArithmeticMeanRank(ArithmeticMeanRank):
    """The adjusted arithmetic mean rank (AMR).

    ---
    description: The mean over all ranks divided by its expected value.
    link: https://arxiv.org/abs/2002.06914
    """

    key = ADJUSTED_ARITHMETIC_MEAN_RANK
    name = "Adjusted Arithmetic Mean Rank (AAMR)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=2, upper_inclusive=False)
    synonyms = ("adjusted_mean_rank", "amr", "aamr")
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True
    increasing = False

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return super().__call__(ranks=ranks) / self.expected_value(num_candidates=num_candidates)


@parse_docdata
class AdjustedArithmeticMeanRankIndex(ArithmeticMeanRank):
    """The adjusted arithmetic mean rank index (AMRI).

    ---
    link: https://arxiv.org/abs/2002.06914
    description: The re-indexed adjusted mean rank (AAMR)
    """

    key = ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX
    name = "Adjusted Arithmetic Mean Rank Index (AAMRI)"
    value_range = ValueRange(lower=-1, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms = ("adjusted_mean_rank_index", "amri", "aamri")
    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True

    def __call__(self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None) -> float:  # noqa: D102
        return 1.0 - (super().__call__(ranks=ranks) - 1.0) / (self.expected_value(num_candidates=num_candidates) - 1.0)


rank_based_metric_resolver: Resolver[RankBasedMetric] = Resolver.from_subclasses(
    base=RankBasedMetric,
    default=InverseHarmonicMeanRank,  # mrr
)

ALL_TYPE_FUNCS: Mapping[str, RankBasedMetric] = {
    ARITHMETIC_MEAN_RANK: ArithmeticMeanRank(),
    HARMONIC_MEAN_RANK: HarmonicMeanRank(),
    GEOMETRIC_MEAN_RANK: GeometricMeanRank(),
    MEDIAN_RANK: MedianRank(),
    INVERSE_ARITHMETIC_MEAN_RANK: InverseArithmeticMeanRank(),
    INVERSE_GEOMETRIC_MEAN_RANK: InverseGeometricMeanRank(),
    INVERSE_HARMONIC_MEAN_RANK: InverseHarmonicMeanRank(),  # This is MRR
    INVERSE_MEDIAN_RANK: InverseMedianRank(),
    # Extra stats stuff
    RANK_STD: StandardDeviation(),
    RANK_VARIANCE: Variance(),
    RANK_MAD: MedianAbsoluteDeviation(),
    RANK_COUNT: Count(),
}


def get_ranking_metrics(ranks: np.ndarray) -> Mapping[str, float]:
    """Calculate all rank-based metrics."""
    rv = {}
    for metric_name, metric_func in ALL_TYPE_FUNCS.items():
        rv[metric_name] = metric_func(ranks)
    return rv
