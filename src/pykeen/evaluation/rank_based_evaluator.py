# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import itertools as itt
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import DefaultDict, Dict, Iterable, List, Literal, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from dataclasses_json import dataclass_json
from scipy import stats

from .evaluator import Evaluator, MetricResults, prepare_filter_triples
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, TARGETS, MappedTriples, Target
from ..utils import fix_dataclass_init_docs

__all__ = [
    "compute_rank_from_scores",
    "RankBasedEvaluator",
    "RankBasedMetricResults",
    "MetricKey",
    "resolve_metric_name",
]

logger = logging.getLogger(__name__)

SideAllType = Literal["both"]
SIDE_BOTH: SideAllType = "both"
SIDES = {LABEL_HEAD, LABEL_TAIL, SIDE_BOTH}
SIDE_BOTH_VALUES = (LABEL_HEAD, LABEL_TAIL)

RANK_OPTIMISTIC = "optimistic"
RANK_PESSIMISTIC = "pessimistic"
RANK_REALISTIC = "realistic"
RANK_TYPES = {RANK_OPTIMISTIC, RANK_PESSIMISTIC, RANK_REALISTIC}

RANK_EXPECTED_REALISTIC = "expected_realistic"
EXPECTED_RANKS = {
    RANK_REALISTIC: RANK_EXPECTED_REALISTIC,
    RANK_OPTIMISTIC: None,  # TODO - research problem
    RANK_PESSIMISTIC: None,  # TODO - research problem
}

ARITHMETIC_MEAN_RANK = "arithmetic_mean_rank"  # also known as mean rank (MR)
GEOMETRIC_MEAN_RANK = "geometric_mean_rank"
HARMONIC_MEAN_RANK = "harmonic_mean_rank"
MEDIAN_RANK = "median_rank"
INVERSE_ARITHMETIC_MEAN_RANK = "inverse_arithmetic_mean_rank"
INVERSE_GEOMETRIC_MEAN_RANK = "inverse_geometric_mean_rank"
INVERSE_HARMONIC_MEAN_RANK = "inverse_harmonic_mean_rank"  # also known as mean reciprocal rank (MRR)
INVERSE_MEDIAN_RANK = "inverse_median_rank"

RANK_STD = "rank_std"
RANK_VARIANCE = "rank_var"
RANK_MAD = "rank_mad"
RANK_COUNT = "rank_count"

all_type_funcs = {
    ARITHMETIC_MEAN_RANK: np.mean,  # This is MR
    HARMONIC_MEAN_RANK: stats.hmean,
    GEOMETRIC_MEAN_RANK: stats.gmean,
    MEDIAN_RANK: np.median,
    INVERSE_ARITHMETIC_MEAN_RANK: lambda x: np.reciprocal(np.mean(x)),
    INVERSE_GEOMETRIC_MEAN_RANK: lambda x: np.reciprocal(stats.gmean(x)),
    INVERSE_HARMONIC_MEAN_RANK: lambda x: np.reciprocal(stats.hmean(x)),  # This is MRR
    INVERSE_MEDIAN_RANK: lambda x: np.reciprocal(np.median(x)),
    # Extra stats stuff
    RANK_STD: np.std,
    RANK_VARIANCE: np.var,
    RANK_MAD: stats.median_abs_deviation,
    RANK_COUNT: lambda x: np.asarray(x.size),
}

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


class MetricKey(NamedTuple):
    """A key for the kind of metric to resolve."""

    name: str
    side: str
    rank_type: str
    k: Optional[int]

    def __str__(self) -> str:  # noqa: D105
        components = [self.name, self.side, self.rank_type]
        if self.k:
            components.append(str(self.k))
        return ".".join(components)


def compute_rank_from_scores(
    true_score: torch.FloatTensor,
    all_scores: torch.FloatTensor,
) -> Dict[str, torch.FloatTensor]:
    """Compute rank and adjusted rank given scores.

    :param true_score: torch.Tensor, shape: (batch_size, 1)
        The score of the true triple.
    :param all_scores: torch.Tensor, shape: (batch_size, num_entities)
        The scores of all corrupted triples (including the true triple).
    :return: a dictionary
        {
            'optimistic': optimistic_rank,
            'pessimistic': pessimistic_rank,
            'realistic': realistic_rank,
            'expected_realistic': expected_realistic_rank,
        }

        where

        optimistic_rank: shape: (batch_size,)
            The optimistic rank is the rank when assuming all options with an equal score are placed behind the current
            test triple.
        pessimistic_rank:
            The pessimistic rank is the rank when assuming all options with an equal score are placed in front of
            current test triple.
        realistic_rank:
            The realistic rank is the average of the optimistic and pessimistic rank, and hence the expected rank
            over all permutations of the elements with the same score as the currently considered option.
        expected_realistic_rank: shape: (batch_size,)
            The expected rank a random scoring would achieve, which is (#number_of_options + 1)/2
    """
    # The optimistic rank is the rank when assuming all options with an equal score are placed behind the currently
    # considered. Hence, the rank is the number of options with better scores, plus one, as the rank is one-based.
    optimistic_rank = (all_scores > true_score).sum(dim=1) + 1

    # The pessimistic rank is the rank when assuming all options with an equal score are placed in front of the
    # currently considered. Hence, the rank is the number of options which have at least the same score minus one
    # (as the currently considered option in included in all options). As the rank is one-based, we have to add 1,
    # which nullifies the "minus 1" from before.
    pessimistic_rank = (all_scores >= true_score).sum(dim=1)

    # The realistic rank is the average of the optimistic and pessimistic rank, and hence the expected rank over
    # all permutations of the elements with the same score as the currently considered option.
    realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

    # We set values which should be ignored to NaN, hence the number of options which should be considered is given by
    number_of_options = torch.isfinite(all_scores).sum(dim=1).float()

    # The expected rank of a random scoring
    expected_realistic_rank = 0.5 * (number_of_options + 1)

    return {
        RANK_OPTIMISTIC: optimistic_rank,
        RANK_PESSIMISTIC: pessimistic_rank,
        RANK_REALISTIC: realistic_rank,
        RANK_EXPECTED_REALISTIC: expected_realistic_rank,
    }


RANK_TYPE_SYNONYMS = {
    "best": RANK_OPTIMISTIC,
    "worst": RANK_PESSIMISTIC,
    "avg": RANK_REALISTIC,
    "average": RANK_REALISTIC,
}

_SIDE_PATTERN = "|".join(SIDES)
_TYPE_PATTERN = "|".join(itt.chain(RANK_TYPES, RANK_TYPE_SYNONYMS.keys()))
METRIC_PATTERN = re.compile(
    rf"(?P<name>[\w@]+)(\.(?P<side>{_SIDE_PATTERN}))?(\.(?P<type>{_TYPE_PATTERN}))?(\.(?P<k>\d+))?",
)
HITS_PATTERN = re.compile(r"(hits_at_|hits@|h@)(?P<k>\d+)")


def resolve_metric_name(name: str) -> MetricKey:
    """Functional metric name normalization."""
    match = METRIC_PATTERN.match(name)
    if not match:
        raise ValueError(f"Invalid metric name: {name}")
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

    return MetricKey(name, side, rank_type, k)


@fix_dataclass_init_docs
@dataclass_json
@dataclass
class RankBasedMetricResults(MetricResults):
    """Results from computing metrics."""

    arithmetic_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Mean Rank (MR)",
            increasing=False,
            range="[1, inf)",
            doc="The arithmetic mean over all ranks.",
            link="https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#mean-rank",
        )
    )

    geometric_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Geometric Mean Rank (GMR)",
            increasing=False,
            range="[1, inf)",
            doc="The geometric mean over all ranks.",
            link="https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html",
        )
    )

    median_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Median Rank",
            increasing=False,
            range="[1, inf)",
            doc="The median over all ranks.",
            link="https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html",
        )
    )

    harmonic_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Harmonic Mean Rank (HMR)",
            increasing=False,
            range="[1, inf)",
            doc="The harmonic mean over all ranks.",
            link="https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html",
        )
    )

    inverse_arithmetic_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Inverse Arithmetic Mean Rank (IAMR)",
            increasing=True,
            range="(0, 1]",
            doc="The inverse of the arithmetic mean over all ranks.",
            link="https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html",
        )
    )

    inverse_geometric_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Inverse Geometric Mean Rank (IGMR)",
            increasing=True,
            range="(0, 1]",
            doc="The inverse of the geometric mean over all ranks.",
            link="https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html",
        )
    )

    inverse_harmonic_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Mean Reciprocal Rank (MRR)",
            increasing=True,
            range="(0, 1]",
            doc="The inverse of the harmonic mean over all ranks.",
            link="https://en.wikipedia.org/wiki/Mean_reciprocal_rank",
        )
    )

    inverse_median_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Inverse Median Rank",
            increasing=True,
            range="(0, 1]",
            doc="The inverse of the median over all ranks.",
            link="https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html",
        )
    )

    rank_count: Dict[str, Dict[str, int]] = field(
        metadata=dict(
            name="Rank Count",
            doc="The number of considered ranks, a non-negative number. Low numbers may indicate unreliable results.",
        )
    )

    rank_std: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Rank Standard Deviation",
            range="[0, inf)",
            increasing=False,
            doc="The standard deviation over all ranks.",
        )
    )

    rank_var: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Rank Variance",
            range="[0, inf)",
            increasing=False,
            doc="The variance over all ranks.",
        )
    )

    rank_mad: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Rank Median Absolute Deviation",
            range="[0, inf)",
            doc="The median absolute deviation over all ranks.",
        )
    )

    hits_at_k: Dict[str, Dict[str, Dict[Union[int, float], float]]] = field(
        metadata=dict(
            name="Hits @ K",
            range="[0, 1]",
            increasing=True,
            doc="The relative frequency of ranks not larger than a given k.",
            link="https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#hits-k",
        )
    )

    adjusted_arithmetic_mean_rank: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Adjusted Arithmetic Mean Rank (AAMR)",
            increasing=False,
            range="(0, 2)",
            doc="The mean over all chance-adjusted ranks.",
            link="https://arxiv.org/abs/2002.06914",
        )
    )

    adjusted_arithmetic_mean_rank_index: Dict[str, Dict[str, float]] = field(
        metadata=dict(
            name="Adjusted Arithmetic Mean Rank Index (AAMRI)",
            increasing=True,
            range="[-1, 1]",
            doc="The re-indexed adjusted mean rank (AAMR)",
            link="https://arxiv.org/abs/2002.06914",
        )
    )

    def get_metric(self, name: str) -> float:
        """Get the rank-based metric.

        :param name: The name of the metric, created by concatenating three parts:

            1. The side (one of "head", "tail", or "both"). Most publications exclusively report "both".
            2. The type (one of "optimistic", "pessimistic", "realistic")
            3. The metric name ("adjusted_mean_rank_index", "adjusted_mean_rank", "mean_rank, "mean_reciprocal_rank",
               "inverse_geometric_mean_rank",
               or "hits@k" where k defaults to 10 but can be substituted for an integer. By default, 1, 3, 5, and 10
               are available. Other K's can be calculated by setting the appropriate variable in the
               ``evaluation_kwargs`` in the :func:`pykeen.pipeline.pipeline` or setting ``ks`` in the
               :class:`pykeen.evaluation.RankBasedEvaluator`.

            In general, all metrics are available for all combinations of sides/types except AMR and AMRI, which
            are only calculated for the average type. This is because the calculation of the expected MR in the
            optimistic and pessimistic case scenarios is still an active area of research and therefore has no
            implementation yet.
        :return: The value for the metric
        :raises ValueError: if an invalid name is given.

        Get the average MR

        >>> metric_results.get('both.realistic.mean_rank')

        If you only give a metric name, it assumes that it's for "both" sides and "realistic" type.

        >>> metric_results.get('adjusted_mean_rank_index')

        This function will do its best to infer what's going on if you only specify one part.

        >>> metric_results.get('left.mean_rank')
        >>> metric_results.get('optimistic.mean_rank')

        Get the default Hits @ K (where $k=10$)

        >>> metric_results.get('hits@k')

        Get a given Hits @ K

        >>> metric_results.get('hits@5')
        """
        metric, side, rank_type, k = resolve_metric_name(name)
        if not metric.startswith("hits"):
            return getattr(self, metric)[side][rank_type]
        assert k is not None
        return self.hits_at_k[side][rank_type][k]

    def to_flat_dict(self):  # noqa: D102
        return {f"{side}.{rank_type}.{metric_name}": value for side, rank_type, metric_name, value in self._iter_rows()}

    def to_df(self) -> pd.DataFrame:
        """Output the metrics as a pandas dataframe."""
        return pd.DataFrame(list(self._iter_rows()), columns=["Side", "Type", "Metric", "Value"])

    def _iter_rows(self) -> Iterable[Tuple[str, str, str, Union[float, int]]]:
        for side, rank_type in itt.product(SIDES, RANK_TYPES):
            for k, v in self.hits_at_k[side][rank_type].items():
                yield side, rank_type, f"hits_at_{k}", v
            for f in fields(self):
                if f.name == "hits_at_k":
                    continue
                side_data = getattr(self, f.name)[side]
                if rank_type in side_data:
                    yield side, rank_type, f.name, side_data[rank_type]


class RankBasedEvaluator(Evaluator):
    r"""A rank-based evaluator for KGE models.

    Calculates the following metrics:

    - Mean Rank (MR) with range $[1, \infty)$ where closer to 0 is better
    - Adjusted Mean Rank (AMR; [berrendorf2020]_) with range $(0, 2)$ where closer to 0 is better
    - Adjusted Mean Rank Index (AMRI; [berrendorf2020]_) with range $[-1, 1]$ where closer to 1 is better
    - Mean Reciprocal Rank (MRR) with range $(0, 1]$ where closer to 1 is better
    - Hits @ K with range $[0, 1]$ where closer to 1 is better.

    .. [berrendorf2020] Berrendorf, *et al.* (2020) `Interpretable and Fair
        Comparison of Link Prediction or Entity Alignment Methods with Adjusted Mean Rank
        <https://arxiv.org/abs/2002.06914>`_.
    """

    ks: Sequence[Union[int, float]]
    num_entities: Optional[int]

    def __init__(
        self,
        ks: Optional[Iterable[Union[int, float]]] = None,
        filtered: bool = True,
        **kwargs,
    ):
        """Initialize rank-based evaluator.

        :param ks:
            The values for which to calculate hits@k. Defaults to {1,3,5,10}.
        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        :param kwargs: Additional keyword arguments that are passed to the base class.
        """
        super().__init__(
            filtered=filtered,
            requires_positive_mask=False,
            **kwargs,
        )
        self.ks = tuple(ks) if ks is not None else (1, 3, 5, 10)
        for k in self.ks:
            if isinstance(k, float) and not (0 < k < 1):
                raise ValueError(
                    "If k is a float, it should represent a relative rank, i.e. a value between 0 and 1 (excl.)",
                )
        self.ranks: Dict[Tuple[Target, str], List[float]] = defaultdict(list)
        self.num_entities = None

    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: Target,
        hrt_batch: MappedTriples,
    ) -> None:
        """Shared code for updating the stored ranks for head/relation/tail scores.

        :param true_scores: shape: (batch_size,)
        :param all_scores: shape: (batch_size, num_entities)
        """
        batch_ranks = compute_rank_from_scores(
            true_score=true_scores,
            all_scores=all_scores,
        )
        self.num_entities = all_scores.shape[1]
        for k, v in batch_ranks.items():
            self.ranks[side, k].extend(v.detach().cpu().tolist())

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=LABEL_TAIL, hrt_batch=hrt_batch)

    def process_relation_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=LABEL_RELATION, hrt_batch=hrt_batch)

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=LABEL_HEAD, hrt_batch=hrt_batch)

    def _get_ranks(self, side: Union[Target, SideAllType], rank_type: str) -> np.ndarray:
        if side == SIDE_BOTH:
            values: List[float] = sum((self.ranks.get((_side, rank_type), []) for _side in TARGETS), [])
        else:
            values = self.ranks.get((side, rank_type), [])
        return np.asarray(values, dtype=np.float64)

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError

        hits_at_k: DefaultDict[str, Dict[str, Dict[Union[int, float], float]]] = defaultdict(dict)
        asr: DefaultDict[str, DefaultDict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for side, rank_type in itt.product(SIDES, RANK_TYPES):
            ranks = self._get_ranks(side=side, rank_type=rank_type)
            if len(ranks) < 1:
                continue
            hits_at_k[side][rank_type] = {
                k: np.mean(ranks <= (k if isinstance(k, int) else int(self.num_entities * k))).item() for k in self.ks
            }
            for metric_name, metric_func in all_type_funcs.items():
                asr[metric_name][side][rank_type] = metric_func(ranks).item()

            expected_rank_type = EXPECTED_RANKS.get(rank_type)
            if expected_rank_type:
                expected_ranks = self._get_ranks(side=side, rank_type=expected_rank_type)
                if 0 < len(expected_ranks):
                    # Adjusted mean rank calculation
                    expected_mean_rank = float(np.mean(expected_ranks))
                    asr[ADJUSTED_ARITHMETIC_MEAN_RANK][side][rank_type] = (
                        asr[ARITHMETIC_MEAN_RANK][side][rank_type] / expected_mean_rank
                    )
                    asr[ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX][side][rank_type] = 1.0 - (
                        asr[ARITHMETIC_MEAN_RANK][side][rank_type] - 1
                    ) / (expected_mean_rank - 1)

        # Clear buffers
        self.ranks.clear()

        # for typing
        rank_count: Dict[str, Dict[str, int]] = dict(asr[RANK_COUNT])  # type: ignore

        return RankBasedMetricResults(
            arithmetic_mean_rank=dict(asr[ARITHMETIC_MEAN_RANK]),
            geometric_mean_rank=dict(asr[GEOMETRIC_MEAN_RANK]),
            harmonic_mean_rank=dict(asr[HARMONIC_MEAN_RANK]),
            median_rank=dict(asr[MEDIAN_RANK]),
            inverse_arithmetic_mean_rank=dict(asr[INVERSE_ARITHMETIC_MEAN_RANK]),
            inverse_geometric_mean_rank=dict(asr[INVERSE_GEOMETRIC_MEAN_RANK]),
            inverse_harmonic_mean_rank=dict(asr[INVERSE_HARMONIC_MEAN_RANK]),
            inverse_median_rank=dict(asr[INVERSE_MEDIAN_RANK]),
            rank_count=rank_count,
            rank_std=dict(asr[RANK_STD]),
            rank_mad=dict(asr[RANK_MAD]),
            rank_var=dict(asr[RANK_VARIANCE]),
            adjusted_arithmetic_mean_rank=dict(asr[ADJUSTED_ARITHMETIC_MEAN_RANK]),
            adjusted_arithmetic_mean_rank_index=dict(asr[ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX]),
            hits_at_k=dict(hits_at_k),
        )


def sample_negatives(
    evaluation_triples: MappedTriples,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    num_samples: int = 50,
    num_entities: Optional[int] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Sample true negatives for sampled evaluation.

    :param evaluation_triples: shape: (n, 3)
        the evaluation triples
    :param additional_filter_triples:
        additional true triples which are to be filtered
    :param num_samples: >0
        the number of samples
    :param num_entities:
        the number of entities

    :return:
        a tuple (id_df, head_negatives, tail_negatives) where

        - id_df: head_id | relation_id | tail_id | index
            a dataframe mapping evaluation triples to their index
        - head_negatives: shape: (n, num_negatives)
            the negatives for head prediction
        - tail_negatives: shape: (n, num_negatives)
            the negatives for tail prediction
    """
    additional_filter_triples = prepare_filter_triples(
        mapped_triples=evaluation_triples,
        additional_filter_triples=additional_filter_triples,
    )
    num_entities = num_entities or (additional_filter_triples[:, [0, 2]].max().item() + 1)
    columns = [LABEL_HEAD, LABEL_RELATION, LABEL_TAIL]
    num_triples = evaluation_triples.shape[0]
    df = pd.DataFrame(data=evaluation_triples.numpy(), columns=columns)
    all_df = pd.DataFrame(data=additional_filter_triples.numpy(), columns=columns)
    id_df = df.reset_index()
    all_ids = set(range(num_entities))
    negatives = []
    for side in [LABEL_HEAD, LABEL_TAIL]:
        this_negatives = torch.empty(size=(num_triples, num_samples), dtype=torch.long)
        other = [c for c in columns if c != side]
        for _, group in pd.merge(id_df, all_df, on=other, suffixes=["_eval", "_all"]).groupby(
            by=other,
        ):
            pool = list(all_ids.difference(group[f"{side}_all"].unique().tolist()))
            if len(pool) < num_samples:
                logger.warning(
                    f"There are less than num_samples={num_samples} candidates for side={side}, triples={group}.",
                )
                # repeat
                pool = int(math.ceil(num_samples / len(pool))) * pool
            for i in group["index"].unique():
                this_negatives[i, :] = torch.as_tensor(
                    data=random.sample(population=pool, k=num_samples),
                    dtype=torch.long,
                )
        negatives.append(this_negatives)
    return negatives[0], negatives[1]


class SampledRankBasedEvaluator(RankBasedEvaluator):
    """
    A rank-based evaluator using sampled negatives instead of all negatives, cf. [teru2020]_.

    Notice that this evaluator yields optimistic estimations of the metrics evaluated on all entities,
    cf. https://arxiv.org/abs/2106.06935.
    """

    def __init__(
        self,
        evaluation_factory: CoreTriplesFactory,
        *,
        additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
        num_negatives: Optional[int] = None,
        head_negatives: Optional[torch.LongTensor] = None,
        tail_negatives: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        :param evaluation_factory:
            the factory with evaluation triples
        :param head_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for head prediction for each evaluation triple
        :param tail_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for tail prediction for each evaluation triple
        :param kwargs:
            additional keyword-based arguments passed to RankBasedEvaluator.__init__
        """
        super().__init__(**kwargs)
        if head_negatives is None and tail_negatives is None:
            # default for inductive LP by [teru2020]
            num_negatives = num_negatives or 50
            logger.info(
                f"Sampling {num_negatives} negatives for each of the "
                f"{evaluation_factory.num_triples} evaluation triples.",
            )
            if num_negatives > evaluation_factory.num_entities:
                raise ValueError("Cannot use more negative samples than there are entities.")
            head_negatives, tail_negatives = sample_negatives(
                evaluation_triples=evaluation_factory.mapped_triples,
                additional_filter_triples=additional_filter_triples,
                num_entities=evaluation_factory.num_entities,
                num_samples=num_negatives,
            )
        elif head_negatives is None or tail_negatives is None:
            raise ValueError("Either both, head and tail negatives must be provided, or none.")

        # verify input
        for negatives in (head_negatives, tail_negatives):
            if negatives.shape[0] != evaluation_factory.num_triples:
                raise ValueError(f"Negatives are in wrong shape: {negatives.shape}")
        self.triple_to_index = {(h, r, t): i for i, (h, r, t) in enumerate(evaluation_factory.mapped_triples.tolist())}
        self.negative_samples: Mapping[Target, torch.FloatTensor] = {
            LABEL_HEAD: head_negatives,
            LABEL_TAIL: tail_negatives,
        }
        self.num_entities = evaluation_factory.num_entities

    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: Target,
        hrt_batch: MappedTriples,
    ) -> None:  # noqa: D102
        # TODO: do not require to compute all scores beforehand
        triple_indices = [self.triple_to_index[h, r, t] for h, r, t in hrt_batch.cpu().tolist()]
        negative_entity_ids = self.negative_samples[side][triple_indices]
        negative_scores = all_scores[
            torch.arange(hrt_batch.shape[0], device=hrt_batch.device).unsqueeze(dim=-1),
            negative_entity_ids,
        ]
        # super.evaluation assumes that the true scores are part of all_scores
        scores = torch.cat([true_scores, negative_scores], dim=-1)
        super()._update_ranks_(true_scores=true_scores, all_scores=scores, side=side, hrt_batch=hrt_batch)
        # write back correct num_entities
        # TODO: should we give num_entities in the constructor instead of inferring it every time ranks are processed?
        self.num_entities = all_scores.shape[1]


def numeric_expected_value(
    metric: str,
    num_candidates: Union[Sequence[int], np.ndarray],
    num_samples: int,
) -> float:
    """
    Compute expected metric value by summation.

    Depending on the metric, the estimate may not be very accurate and converage slowly, cf.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
    """
    metric_func = all_type_funcs[metric]
    num_candidates = np.asarray(num_candidates)
    generator = np.random.default_rng()
    expectation = 0
    for _ in range(num_samples):
        ranks = generator.integers(low=0, high=num_candidates)
        expectation += metric_func(ranks)
    return expectation / num_samples


# TODO: closed-forms for other metrics?


def expected_mean_rank(
    num_candidates: Union[Sequence[int], np.ndarray],
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


def expected_hits_at_k(
    num_candidates: Union[Sequence[int], np.ndarray],
    k: int,
) -> float:
    r"""
    Calculate the expected Hits@k under random ordering.

    .. math ::

        E[Hits@k] = \frac{1}{n} \sum \limits_{i=1}^{n} min(\frac{k}{CSS[i]}, 1.0)

    :param num_candidates:
        the number of candidates for each individual rank computation

    :return:
        the expected Hits@k value
    """
    return k * np.mean(np.reciprocal(np.asanyarray(num_candidates, dtype=float)).clip(min=None, max=1 / k))
