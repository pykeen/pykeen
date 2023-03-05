# -*- coding: utf-8 -*-

# todo: deprecated

"""Lookup for metrics."""

import itertools as itt
import logging
import re
import warnings
from typing import Any, Mapping, NamedTuple, Optional, Tuple, Union, cast

from class_resolver import ClassResolver

from ..metrics import Metric
from ..metrics.ranking import HitsAtK, InverseHarmonicMeanRank, RankBasedMetric, rank_based_metric_resolver
from ..typing import RANK_REALISTIC, RANK_TYPE_SYNONYMS, RANK_TYPES, SIDE_BOTH, SIDES, ExtendedTarget, RankType
from ..utils import flatten_dictionary

__all__ = [
    "MetricKey",
]

logger = logging.getLogger(__name__)

# parsing metrics
# metric pattern = side?.type?.metric.k?
_SIDE_PATTERN = "|".join(SIDES)
_TYPE_PATTERN = "|".join(itt.chain(RANK_TYPES, RANK_TYPE_SYNONYMS.keys()))
METRIC_PATTERN = re.compile(
    rf"^((?P<side>{_SIDE_PATTERN})\.)?((?P<type>{_TYPE_PATTERN})\.)?(?P<name>[\w@]+)(\.(?P<k>\d+))?$",
)
HITS_PATTERN = re.compile(r"(?P<name>h@|hits@|hits_at_)(?P<k>\d+)")


class MetricKey(NamedTuple):
    """A key for the kind of metric to resolve."""

    #: The metric key
    metric: str

    #: Side of the metric, or "both"
    side: ExtendedTarget

    #: The rank type
    rank_type: RankType

    def __str__(self) -> str:  # noqa: D105
        return ".".join(map(str, (self.side, self.rank_type, self.metric)))

    @classmethod
    def lookup(
        cls, s: Union[None, str, Tuple[str, ExtendedTarget, RankType]], resolver: Optional[ClassResolver[Metric]] = None
    ) -> "MetricKey":
        """Functional metric name normalization."""
        # TODO: the "rank-type" part does not make much sense for non-rank-based metrics
        if isinstance(s, tuple):
            return cls.lookup(str(MetricKey(*s)))

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
        if resolver is None:
            warnings.warn(
                f"Using {cls.__name__}.lookup without providing a `resolver` is deprecated.",
                category=DeprecationWarning,
            )
            resolver = rank_based_metric_resolver
        metric_cls = resolver.lookup(name)

        kwargs = {}
        if issubclass(metric_cls, HitsAtK):
            k = int(k or 10)
            assert k > 0
            kwargs["k"] = k

        metric = resolver.make(metric_cls, kwargs)

        # normalize side
        side = side or SIDE_BOTH
        side = side.lower()
        if side not in SIDES:
            raise ValueError(f"Invalid side: {side}. Allowed are {SIDES}.")

        # fixme: non-rank-based metrics do not have a rank-type
        if isinstance(metric, RankBasedMetric):
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
        else:
            rank_type = RANK_REALISTIC

        return cls(metric.key, cast(ExtendedTarget, side), cast(RankType, rank_type))

    @classmethod
    def normalize(cls, s: Optional[str], resolver: Optional[ClassResolver[Metric]] = None) -> str:
        """Normalize a metric key string."""
        return str(cls.lookup(s, resolver=resolver))


def normalize_flattened_metric_results(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Flatten metric result dictionary and normalize metric keys.

    :param result:
        the result dictionary.

    :return:
        the flattened metric results with normalized metric names.
    """
    # normalize keys
    # TODO: this can only normalize rank-based metrics!
    # TODO: find a better way to handle this
    flat_result = flatten_dictionary(result)
    result = {}
    for key, value in flat_result.items():
        try:
            key = MetricKey.normalize(key)
        except ValueError as error:
            logger.warning(f"Trying to fix malformed key: {error}")
            key = MetricKey.normalize(
                key.replace("nondeterministic", "").replace("unknown", "").strip(".").replace("..", ".")
            )
        result[key] = value
    return result
