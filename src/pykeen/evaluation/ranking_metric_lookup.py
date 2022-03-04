# -*- coding: utf-8 -*-

"""Lookup for metrics."""

import itertools as itt
import re
from typing import NamedTuple, Optional, Union, cast

from ..metrics.ranking import HitsAtK, InverseHarmonicMeanRank, rank_based_metric_resolver
from ..typing import RANK_REALISTIC, RANK_TYPE_SYNONYMS, RANK_TYPES, SIDE_BOTH, SIDES, ExtendedRankType, ExtendedTarget

__all__ = [
    "RankingMetricKey",
]

# parsing metrics
# metric pattern = side?.type?.metric.k?
_SIDE_PATTERN = "|".join(SIDES)
_TYPE_PATTERN = "|".join(itt.chain(RANK_TYPES, RANK_TYPE_SYNONYMS.keys()))
METRIC_PATTERN = re.compile(
    rf"^((?P<side>{_SIDE_PATTERN})\.)?((?P<type>{_TYPE_PATTERN})\.)?(?P<name>[\w@]+)(\.(?P<k>\d+))?$",
)
HITS_PATTERN = re.compile(r"(?P<name>h@|hits@|hits_at_)(?P<k>\d+)")


class RankingMetricKey(NamedTuple):
    """A key for the kind of metric to resolve."""

    #: The metric key
    metric: str

    #: Side of the metric, or "both"
    side: ExtendedTarget

    #: The rank type
    rank_type: ExtendedRankType

    def __str__(self) -> str:  # noqa: D105
        return ".".join(map(str, (self.side, self.rank_type, self.metric)))

    @classmethod
    def lookup(cls, s: Optional[str]) -> "RankingMetricKey":
        """Functional metric name normalization."""
        if s is None:
            return cls(metric=InverseHarmonicMeanRank().key, side=SIDE_BOTH, rank_type=RANK_REALISTIC)

        match = METRIC_PATTERN.match(s)
        if not match:
            raise ValueError(f"Invalid metric name: {s}")
        k: Union[None, str, int]
        name, side, rank_type, k = [match.group(key) for key in ("name", "side", "type", "k")]
        name = name.lower()
        match = HITS_PATTERN.match(name)
        if match:
            name, k = match.groups()

        # normalize metric name
        if not name:
            raise ValueError("A metric name must be provided.")
        metric_cls = rank_based_metric_resolver.lookup(name)

        kwargs = {}
        if issubclass(metric_cls, HitsAtK):
            k = int(k or 10)
            assert k > 0
            kwargs["k"] = k

        metric = rank_based_metric_resolver.make(metric_cls, kwargs)

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
        elif rank_type not in metric.supported_rank_types:
            raise ValueError(
                f"Invalid rank type for {metric}: {rank_type}. Allowed type: {metric.supported_rank_types}"
            )

        return cls(metric.key, cast(ExtendedTarget, side), cast(ExtendedRankType, rank_type))

    @classmethod
    def normalize(cls, s: Optional[str]) -> str:
        """Normalize a metric key string."""
        return str(cls.lookup(s))
