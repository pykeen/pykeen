# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from dataclasses_json import dataclass_json

from .evaluator import Evaluator, MetricResults
from ..typing import MappedTriples
from ..utils import fix_dataclass_init_docs

__all__ = [
    'compute_rank_from_scores',
    'RankBasedEvaluator',
    'RankBasedMetricResults',
]

logger = logging.getLogger(__name__)

SIDE_HEAD = 'head'
SIDE_TAIL = 'tail'
SIDE_BOTH = 'both'
RANK_BEST = 'best'
RANK_WORST = 'worst'
RANK_AVERAGE = 'avg'
RANK_EXPECTED = "exp"
RANK_TYPES = {RANK_BEST, RANK_WORST, RANK_AVERAGE}
SIDES = {SIDE_HEAD, SIDE_TAIL, SIDE_BOTH}

MEAN_RANK = 'mean_rank'
MEAN_RECIPROCAL_RANK = 'mean_reciprocal_rank'
ADJUSTED_MEAN_RANK = 'adjusted_mean_rank'
ADJUSTED_MEAN_RANK_INDEX = 'adjusted_mean_rank_index'
TYPES_ALL = {MEAN_RANK, MEAN_RECIPROCAL_RANK}
TYPES_AVG_ONLY = {ADJUSTED_MEAN_RANK, ADJUSTED_MEAN_RANK_INDEX}


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
            'best': best_rank,
            'worst': worst_rank,
            'avg': avg_rank,
            'exp': exp_rank,
        }

        where

        best_rank: shape: (batch_size,)
            The best rank is the rank when assuming all options with an equal score are placed behind the current
            test triple.
        worst_rank:
            The worst rank is the rank when assuming all options with an equal score are placed in front of current
            test triple.
        avg_rank:
            The average rank is the average of the best and worst rank, and hence the expected rank over all
            permutations of the elements with the same score as the currently considered option.
        exp_rank: shape: (batch_size,)
            The expected rank a random scoring would achieve, which is (#number_of_options + 1)/2
    """
    # The best rank is the rank when assuming all options with an equal score are placed behind the currently
    # considered. Hence, the rank is the number of options with better scores, plus one, as the rank is one-based.
    best_rank = (all_scores > true_score).sum(dim=1) + 1

    # The worst rank is the rank when assuming all options with an equal score are placed in front of the currently
    # considered. Hence, the rank is the number of options which have at least the same score minus one (as the
    # currently considered option in included in all options). As the rank is one-based, we have to add 1, which
    # nullifies the "minus 1" from before.
    worst_rank = (all_scores >= true_score).sum(dim=1)

    # The average rank is the average of the best and worst rank, and hence the expected rank over all permutations of
    # the elements with the same score as the currently considered option.
    average_rank = (best_rank + worst_rank).float() * 0.5

    # We set values which should be ignored to NaN, hence the number of options which should be considered is given by
    number_of_options = torch.isfinite(all_scores).sum(dim=1).float()

    # The expected rank of a random scoring
    expected_rank = 0.5 * (number_of_options + 1)

    return {
        RANK_BEST: best_rank,
        RANK_WORST: worst_rank,
        RANK_AVERAGE: average_rank,
        RANK_EXPECTED: expected_rank,
    }


@fix_dataclass_init_docs
@dataclass_json
@dataclass
class RankBasedMetricResults(MetricResults):
    r"""Results from computing metrics."""

    mean_rank: Dict[str, Dict[str, float]] = field(metadata=dict(
        name="Mean Rank (MR)",
        doc='The mean over all ranks on, [1, inf). Lower is better.',
    ))

    mean_reciprocal_rank: Dict[str, Dict[str, float]] = field(metadata=dict(
        name="Mean Reciprocal Rank (MRR)",
        doc='The mean over all reciprocal ranks, on (0, 1]. Higher is better.',
    ))

    hits_at_k: Dict[str, Dict[str, Dict[Union[int, float], float]]] = field(metadata=dict(
        name='Hits @ K',
        doc='The relative frequency of ranks not larger than a given k, on [0, 1]. Higher is better',
    ))

    adjusted_mean_rank: Dict[str, float] = field(metadata=dict(
        name='Adjusted Mean Rank (AMR)',
        doc='The mean over all chance-adjusted ranks, on (0, 2). Lower is better.',
    ))

    adjusted_mean_rank_index: Dict[str, float] = field(metadata=dict(
        name='Adjusted Mean Rank Index (AMRI)',
        doc='The re-indexed adjusted mean rank (AMR), on [-1, 1]. Higher is better.',
    ))

    def __post_init__(self):  # noqa:D105
        self._types_avg_only = {
            ADJUSTED_MEAN_RANK: self.adjusted_mean_rank,
            ADJUSTED_MEAN_RANK_INDEX: self.adjusted_mean_rank_index,
        }
        self._types_all = {
            MEAN_RANK: self.mean_rank,
            MEAN_RECIPROCAL_RANK: self.mean_reciprocal_rank,
        }

    def get_metric(self, name: str) -> float:
        """Get the rank-based metric.

        :param name: The name of the metric, created by concatenating three parts:

            1. The side ("head", "tail", or "both"). Most publications exclusively report "both".
            2. The type ("avg", "best", "worst")
            3. The metric name ("adjusted_mean_rank_index", "adjusted_mean_rank", "mean_rank, "mean_reciprocal_rank"
               or "hits@k" where k defaults to 10 but can be substituted for an integer. By default, 1, 3, 5, and 10
               are available. Other K's can be calculated by setting the appropriate variable in the
               ``evaluation_kwargs`` in the :func:`pykeen.pipeline.pipeline` or setting ``ks`` in the
               :class:`pykeen.evaluation.RankBasedEvaluator`.

            In general, all metrics are available for all combinations of sides/types except AMR and AMRI, which
            are only calculated for the average type. This is because the calculation of the expected MR in the
            best and worst case scenarios is still an active area of research and therefore has no implementation yet.
        :return: The value for the metric
        :raises ValueError: if an invalid name is given.

        Get the average MR

        >>> metric_results.get('both.avg.mean_rank')

        If you only give a metric name, it assumes that it's for "both" sides and "average" type.

        >>> metric_results.get('adjusted_mean_rank_index')

        This function will do its best to infer what's going on if you only specify one part.

        >>> metric_results.get('left.mean_rank')
        >>> metric_results.get('best.mean_rank')

        Get the default Hits @ K (where $k=10$)

        >>> metric_results.get('hits@k')

        Get a given Hits @ K

        >>> metric_results.get('hits@5')
        """
        dot_count = name.count('.')
        if 0 == dot_count:  # assume average by default
            side, rank_type, metric = SIDE_BOTH, RANK_AVERAGE, name
        elif 1 == dot_count:
            # Check if it a side or rank type
            side_or_ranktype, metric = name.split('.')
            if side_or_ranktype in SIDES:
                side = side_or_ranktype
                rank_type = RANK_AVERAGE
            else:
                side = SIDE_BOTH
                rank_type = side_or_ranktype
        elif 2 == dot_count:
            side, rank_type, metric = name.split('.')
        else:
            raise ValueError(f'Malformed metric name: {name}')

        if side not in SIDES:
            raise ValueError(f'Invalid side: {side}. Allowed sides: {SIDES}')
        if rank_type not in RANK_AVERAGE and metric in TYPES_AVG_ONLY:
            raise ValueError(f'Invalid rank type for {metric}: {rank_type}. Allowed type: {RANK_AVERAGE}')
        elif rank_type not in RANK_TYPES:
            raise ValueError(f'Invalid rank type: {rank_type}. Allowed types: {RANK_TYPES}')

        if metric in TYPES_ALL:
            return getattr(self, metric)[side][rank_type]
        elif metric in TYPES_AVG_ONLY:
            return getattr(self, metric)[side]

        # otherwise, assume is hits@k, which is handled differently
        rank_type_hits_at_k = self.hits_at_k[side][rank_type]
        for prefix in ('hits_at_', 'hits@'):
            if not metric.startswith(prefix):
                continue
            k = metric[len(prefix):]
            k_int = 10 if k == 'k' else int(k)
            return rank_type_hits_at_k[k_int]

        raise ValueError(f'Invalid metric name: {name}')

    def to_flat_dict(self):  # noqa: D102
        return {
            f'{side}.{rank_type}.{metric_name}': value
            for side, rank_type, metric_name, value in self._iter_rows()
        }

    def to_df(self) -> pd.DataFrame:
        """Output the metrics as a pandas dataframe."""
        return pd.DataFrame(list(self._iter_rows()), columns=['Side', 'Type', 'Metric', 'Value'])

    def _iter_rows(self) -> Iterable[Tuple[str, str, str, float]]:
        for side in SIDES:
            for metric_name, metric_dict in sorted(self._types_avg_only.items()):
                yield side, RANK_AVERAGE, metric_name, metric_dict[side]
            for rank_type in RANK_TYPES:
                for metric_name, metric_dict in sorted(self._types_all.items()):
                    yield side, rank_type, metric_name, metric_dict[side][rank_type]
                for k, v in self.hits_at_k[side][rank_type].items():
                    yield side, rank_type, f'hits_at_{k}', v


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

    def __init__(
        self,
        ks: Optional[Iterable[Union[int, float]]] = None,
        filtered: bool = True,
        automatic_memory_optimization: bool = True,
        **kwargs,
    ):
        """Initialize rank-based evaluator.

        :param ks:
            The values for which to calculate hits@k. Defaults to {1,3,5,10}.
        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        """
        super().__init__(filtered=filtered, automatic_memory_optimization=automatic_memory_optimization, **kwargs)
        self.ks = tuple(ks) if ks is not None else (1, 3, 5, 10)
        for k in self.ks:
            if isinstance(k, float) and not (0 < k < 1):
                raise ValueError(
                    'If k is a float, it should represent a relative rank, i.e. a value between 0 and 1 (excl.)',
                )
        self.ranks: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.num_entities = None

    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: str,
    ) -> None:
        """Shared code for updating the stored ranks for head/tail scores.

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
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=SIDE_TAIL)

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=SIDE_HEAD)

    def _get_ranks(self, side, rank_type) -> np.ndarray:
        if side == SIDE_BOTH:
            values: List[float] = sum((self.ranks.get((_side, rank_type), []) for _side in (SIDE_HEAD, SIDE_TAIL)), [])
        else:
            values = self.ranks.get((side, rank_type), [])
        return np.asarray(values, dtype=np.float64)

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        mean_rank: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        mean_reciprocal_rank: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        hits_at_k: DefaultDict[str, Dict[str, Dict[Union[int, float], float]]] = defaultdict(dict)
        adjusted_mean_rank: Dict[str, float] = {}
        adjusted_mean_rank_index: Dict[str, float] = {}

        if self.num_entities is None:
            raise ValueError

        for side in SIDES:
            for rank_type in RANK_TYPES:
                ranks = self._get_ranks(side=side, rank_type=rank_type)
                if len(ranks) < 1:
                    continue
                hits_at_k[side][rank_type] = {
                    k: np.mean(ranks <= (k if isinstance(k, int) else int(self.num_entities * k)))
                    for k in self.ks
                }
                mean_rank[side][rank_type] = np.mean(ranks)
                mean_reciprocal_rank[side][rank_type] = np.mean(np.reciprocal(ranks))

            expected_ranks = self._get_ranks(side=side, rank_type=RANK_EXPECTED)
            if len(expected_ranks) < 1:
                continue
            expected_mean_rank = float(np.mean(expected_ranks))
            adjusted_mean_rank[side] = mean_rank[side][RANK_AVERAGE] / expected_mean_rank
            adjusted_mean_rank_index[side] = 1.0 - (mean_rank[side][RANK_AVERAGE] - 1) / (expected_mean_rank - 1)

        # Clear buffers
        self.ranks.clear()

        return RankBasedMetricResults(
            mean_rank=dict(mean_rank),
            mean_reciprocal_rank=dict(mean_reciprocal_rank),
            hits_at_k=dict(hits_at_k),
            adjusted_mean_rank=adjusted_mean_rank,
            adjusted_mean_rank_index=adjusted_mean_rank_index,
        )
