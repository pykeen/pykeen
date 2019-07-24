# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
import timeit
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from dataclasses_json import dataclass_json
from tqdm import tqdm

from .base import Evaluator
from ..models.base import BaseModule

__all__ = [
    'MetricResults',
    'RankBasedEvaluator',
]

log = logging.getLogger(__name__)


@dataclass_json
@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    mean_reciprocal_rank: float
    adjusted_mean_rank: float
    adjusted_mean_reciprocal_rank: float
    hits_at_k: Dict[int, float]


def _compute_rank_from_scores(true_score, all_scores) -> Tuple[int, float]:
    """Compute rank and adjusted rank given scores.

    :param true_score: The score of the true triple.
    :param all_scores: The scores of all corrupted triples.
    :return: a tuple (best_rank, adjusted_avg_rank) where
        best_rank: int (best_rank >= 1)
            The rank of the true triple as given as the number of elements having a better score plus one.
        adjusted_avg_rank: float (adjusted_avg_rank > 0)
            The avg rank of the true triple divided by the expected rank in random scoring.
    """
    best_rank = (all_scores > true_score).sum(dim=1) + 1
    worst_rank = (all_scores >= true_score).sum(dim=1) + 1
    avg_rank = (best_rank + worst_rank) / 2.0
    adjusted_avg_rank = avg_rank / ((all_scores.shape[0] + 1) / 2)
    return best_rank.detach().cpu().numpy(), adjusted_avg_rank.detach().cpu().numpy()

class RankBasedEvaluator(Evaluator):

    def __init__(
            self,
            model: BaseModule = None,
            filter_neg_triples: bool = False,
            hits_at_k: Optional[List[int]] = None,
    ) -> None:
        super().__init__(model=model)
        self.filter_neg_triples = filter_neg_triples
        self.hits_at_k = hits_at_k if hits_at_k is not None else [1, 3, 5, 10]

    @property
    def train_triples(self):
        return self.model.triples_factory.triples

    @property
    def all_entities(self):
        return self.model.triples_factory.all_entities

    @staticmethod
    def _hash_triples(triples: Iterable[Hashable]) -> int:
        """Hash a list of triples."""
        return hash(tuple(triples))

    def _filter_corrupted_triples(
            self,
            pos_triple,
            subject_batch,
            object_batch,
            all_pos_triples,
    ):
        subject = pos_triple[0:1]
        relation = pos_triple[1:2]
        object = pos_triple[2:3]

        subject_filter = all_pos_triples[:, 0:1] == subject
        relation_filter = all_pos_triples[:, 1:2] == relation
        object_filter = all_pos_triples[:, 2:3] == object

        # Short objects batch list
        filter = (subject_filter & relation_filter)
        objects_in_triples = all_pos_triples[:, 2:3][filter]
        object_batch[objects_in_triples] = 0

        # Short subjects batch list
        filter = (object_filter & relation_filter)
        subjects_in_triples = all_pos_triples[:, 0:1][filter]
        subject_batch[subjects_in_triples] = 0

        # TODO: Create warning when all triples will be filtered
        # if mask.size == 0:
        #     raise Exception("User selected filtered metric computation, but all corrupted triples exists"
        #                     "also as positive triples.")

        return subject_batch, object_batch

    def _update_hits_at_k(
            self,
            hits_at_k_values: Dict[int, List[float]],
            rank_of_positive_subject_based: int,
            rank_of_positive_object_based: int,
    ) -> None:
        """Update the Hits@K dictionary for two values."""
        for k, values in hits_at_k_values.items():
            if rank_of_positive_subject_based <= k:
                values.append(1.0)
            else:
                values.append(0.0)

            if rank_of_positive_object_based <= k:
                values.append(1.0)
            else:
                values.append(0.0)

    def _compute_filtered_rank(
            self,
            pos_triple,
            subject_batch,
            object_batch,
            all_pos_triples,
    ) -> Tuple[int, int, float, float]:
        subject_batch, object_batch = self._filter_corrupted_triples(
            pos_triple=pos_triple,
            subject_batch=subject_batch,
            object_batch=object_batch,
            all_pos_triples=all_pos_triples,
        )

        return self._compute_rank(
            pos_triple=pos_triple,
            subject_batch=subject_batch,
            object_batch=object_batch,
            all_pos_triples=all_pos_triples,
        )

    def _compute_rank(
            self,
            pos_triple,
            subject_batch,
            object_batch,
            all_pos_triples,
    ) -> Tuple[int, int, float, float]:
        subject = pos_triple[0:1]
        object = pos_triple[2:3]

        scores_of_corrupted_subjects = self.model.predict_scores_all_subjects(pos_triple[1:3])
        score_of_positive_subject = scores_of_corrupted_subjects[:, subject]
        scores_of_corrupted_subjects = scores_of_corrupted_subjects[:, subject_batch]

        scores_of_corrupted_objects = self.model.predict_scores_all_objects(pos_triple[0:2])
        score_of_positive_object = scores_of_corrupted_objects[:, object]
        scores_of_corrupted_objects = scores_of_corrupted_objects[:, object_batch]

        rank_of_positive_subject_based, adj_rank_of_positive_subject_based = _compute_rank_from_scores(
            true_score=score_of_positive_subject, all_scores=scores_of_corrupted_subjects,
        )
        rank_of_positive_object_based, adj_rank_of_positive_object_based = _compute_rank_from_scores(
            true_score=score_of_positive_object, all_scores=scores_of_corrupted_objects,
        )

        return (
            rank_of_positive_subject_based,
            rank_of_positive_object_based,
            adj_rank_of_positive_subject_based,
            adj_rank_of_positive_object_based,
        )

    def evaluate(self, test_triples: np.ndarray, use_tqdm: bool = True) -> MetricResults:
        start = timeit.default_timer()
        ranks: List[int] = []
        adj_ranks = np.empty(shape=(test_triples.shape[0], 2), dtype=np.float)
        hits_at_k_values = {
            k: []
            for k in self.hits_at_k
        }
        # Set eval mode in order to ignore functionalities such as dropout
        self.model = self.model.eval()

        all_pos_triples = np.concatenate([self.model.triples_factory.mapped_triples, test_triples], axis=0)
        all_pos_triples = torch.tensor(all_pos_triples, device=self.device)
        all_entities = torch.tensor(self.model.triples_factory.all_entities, device=self.device)

        test_triples = torch.tensor(test_triples, dtype=torch.long, device=self.device)

        compute_rank_fct: Callable[..., Tuple[int, int, float, float]] = (
            self._compute_filtered_rank
            if self.filter_neg_triples else
            self._compute_rank
        )

        if use_tqdm:
            test_triples = tqdm(test_triples, desc=f'Evaluating triples')

        for i, pos_triple in enumerate(test_triples):
            subject = pos_triple[0:1]
            object = pos_triple[2:3]
            subject_batch = all_entities != subject
            object_batch = all_entities != object

            # Disable gradient tracking
            with torch.no_grad():
                (
                    rank_of_positive_subject_based,
                    rank_of_positive_object_based,
                    adjusted_rank_of_positive_subject_based,
                    adjusted_rank_of_positive_object_based,
                ) = compute_rank_fct(
                    pos_triple=pos_triple,
                    subject_batch=subject_batch,
                    object_batch=object_batch,
                    all_pos_triples=all_pos_triples,
                )

            ranks.append(rank_of_positive_subject_based)
            ranks.append(rank_of_positive_object_based)
            adj_ranks[i, :] = (adjusted_rank_of_positive_subject_based, adjusted_rank_of_positive_object_based)

            # Compute hits@k for k in {1,3,5,10}
            self._update_hits_at_k(
                hits_at_k_values,
                rank_of_positive_subject_based=rank_of_positive_subject_based,
                rank_of_positive_object_based=rank_of_positive_object_based,
            )

        mean_rank = float(np.mean(ranks))
        mean_reciprocal_rank = float(np.mean(np.reciprocal(ranks)))
        adjusted_mean_rank = float(np.mean(adj_ranks))
        adjusted_mean_reciprocal_rank = float(np.mean(np.reciprocal(adj_ranks)))
        hits_at_k: Dict[int, float] = {
            k: np.mean(values)
            for k, values in hits_at_k_values.items()
        }

        stop = timeit.default_timer()
        log.info("Evaluation took %.2fs seconds", stop - start)

        return MetricResults(
            mean_rank=mean_rank,
            mean_reciprocal_rank=mean_reciprocal_rank,
            hits_at_k=hits_at_k,
            adjusted_mean_rank=adjusted_mean_rank,
            adjusted_mean_reciprocal_rank=adjusted_mean_reciprocal_rank,
        )
