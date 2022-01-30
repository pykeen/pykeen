# -*- coding: utf-8 -*-

"""Utilities for metrics."""

import itertools as itt
import re
from typing import NamedTuple, Optional, Union, cast

from ..typing import RANK_REALISTIC, RANK_TYPE_SYNONYMS, RANK_TYPES, SIDE_BOTH, SIDES, RankType

__all__ = [
    "MetricKey",
]

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
    rf"(?P<name>[\w@]+)(\.(?P<side>{_SIDE_PATTERN}))?(\.(?P<type>{_TYPE_PATTERN}))?(\.(?P<k>\d+))?",
)
HITS_PATTERN = re.compile(r"(hits_at_|hits@|h@)(?P<k>\d+)")


class MetricKey(NamedTuple):
    """A key for the kind of metric to resolve."""

    #: Name of the metric
    name: str
    #: Side of the metric, or "both"
    side: str
    #: The rank type
    rank_type: RankType
    #: The k if this represents a hits at k metric
    k: Optional[int]

    def __str__(self) -> str:  # noqa: D105
        components = [self.name, self.side, self.rank_type]
        if self.k:
            components.append(str(self.k))
        return ".".join(components)

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

        return cls(name, side, cast(RankType, rank_type), k)

    @classmethod
    def normalize(cls, s: str) -> str:
        """Normalize a metric key string."""
        return str(cls.lookup(s))
