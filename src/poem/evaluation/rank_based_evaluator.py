# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from dataclasses_json import dataclass_json

from .evaluator import Evaluator, MetricResults
from ..typing import MappedTriples

__all__ = [
    'compute_rank_from_scores',
    'RankBasedEvaluator',
    'RankBasedMetricResults',
]

logger = logging.getLogger(__name__)


def compute_rank_from_scores(
    true_score: torch.FloatTensor,
    all_scores: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute rank and adjusted rank given scores.

    :param true_score: torch.Tensor, shape: (batch_size, 1)
        The score of the true triple.
    :param all_scores: torch.Tensor, shape: (batch_size, num_entities)
        The scores of all corrupted triples (including the true triple).

    :return: a tuple (avg_rank, adjusted_avg_rank) where
        avg_rank: shape: (batch_size,)
            The average rank is the average of the best and worst rank, and hence the expected rank over all
            permutations of the elements with the same score as the currently considered option.
        adjusted_avg_rank: shape: (batch_size,)
            The adjusted average rank normalises the average rank by the expected rank a random scoring would achieve,
            which is (#number_of_options + 1)/2
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
    avg_rank = (best_rank + worst_rank).float() * 0.5

    # The adjusted average rank normalises the average rank by the expected rank a random scoring would achieve, which
    # is (#number_of_options + 1)/2

    # We set values which should be ignored to NaN, hence the number of options which should be considered is given by
    number_of_options = torch.isfinite(all_scores).sum(dim=1).float()

    adjusted_avg_rank = avg_rank / ((number_of_options + 1) * 0.5)

    return avg_rank, adjusted_avg_rank


@dataclass_json
@dataclass
class RankBasedMetricResults(MetricResults):
    """Results from computing metrics."""

    #: The mean over all ranks: mean_i r_i
    mean_rank: float = field(metadata=dict(doc='The mean over all ranks: mean_i r_i'))

    #: The mean over all reciprocal ranks: mean_i (1/r_i)
    mean_reciprocal_rank: float = field(metadata=dict(doc='The mean over all reciprocal ranks: mean_i (1/r_i)'))

    #: The mean over all chance-adjusted ranks: mean_i (2r_i / (num_entities+1))
    adjusted_mean_rank: float = field(
        metadata=dict(doc='The mean over all chance-adjusted ranks: mean_i (2r_i / (num_entities+1))'))

    #: The hits at k for different values of k, i.e. the relative frequency of ranks not larger than k
    hits_at_k: Dict[int, float] = field(metadata=dict(
        doc='The hits at k for different values of k, i.e. the relative frequency of ranks not larger than k'))


class RankBasedEvaluator(Evaluator):
    """A rank-based evaluator for KGE models."""

    def __init__(
        self,
        ks: Optional[Iterable[int]] = None,
        filtered: bool = True,
    ):
        super().__init__(filtered=filtered)
        self.ks = tuple(ks) if ks is not None else (1, 3, 5, 10)
        self.ranks: List[float] = []
        self.adj_ranks: List[float] = []

    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
    ) -> None:
        """Shared code for updating the stored ranks for head/tail scores.

        :param true_scores: shape: (batch_size,)
        :param all_scores: shape: (batch_size, num_entities)
        """
        rank, adj_rank = compute_rank_from_scores(true_score=true_scores, all_scores=all_scores)
        self.ranks.extend(rank.detach().cpu().numpy())
        self.adj_ranks.extend(adj_rank.detach().cpu().numpy())

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.BoolTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores)

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.BoolTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores)

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        ranks = np.asarray(self.ranks, dtype=np.float64)
        hits_at_k = {
            k: np.mean(ranks <= k) for k in self.ks
        }
        mr = np.mean(ranks)
        mrr = np.mean(np.reciprocal(ranks))

        adj_ranks = np.asarray(self.adj_ranks, dtype=np.float64)
        amr = np.mean(adj_ranks)

        return RankBasedMetricResults(
            mean_rank=mr,
            mean_reciprocal_rank=mrr,
            hits_at_k=hits_at_k,
            adjusted_mean_rank=amr,
        )
