# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
import timeit
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from dataclasses_json import dataclass_json

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
    best_rank = np.greater(all_scores, true_score).sum() + 1
    worst_rank = np.greater_equal(all_scores, true_score).sum() + 1
    avg_rank = (best_rank + worst_rank) / 2.0
    adjusted_avg_rank = avg_rank / ((all_scores.shape[0] + 1) / 2)
    return best_rank, adjusted_avg_rank


class RankBasedEvaluator(Evaluator):

    def __init__(
            self,
            model: BaseModule,
            entity_to_id,
            relation_to_id,
            training_triples: np.ndarray,
            filter_neg_triples=False,
            hits_at_k: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            model=model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        self.all_entities = np.arange(0, len(self.entity_to_id))
        self.filter_neg_triples = filter_neg_triples
        self.hits_at_k = hits_at_k if hits_at_k is not None else [1, 3, 5, 10]
        self.train_triples = training_triples

    def _hash_triples(self, triples: Iterable[Hashable]) -> int:
        """Hash a list of triples."""
        return hash(tuple(triples))

    def _filter_corrupted_triples(
            self,
            corrupted_subject_based,
            corrupted_object_based,
            all_pos_triples_hashed,
    ):
        # TODO: Check
        corrupted_subject_based_hashed = np.apply_along_axis(self._hash_triples, 1, corrupted_subject_based)
        mask = np.in1d(corrupted_subject_based_hashed, all_pos_triples_hashed, invert=True)
        mask = np.where(mask)[0]
        corrupted_subject_based = corrupted_subject_based[mask]

        corrupted_object_based_hashed = np.apply_along_axis(self._hash_triples, 1, corrupted_object_based)
        mask = np.in1d(corrupted_object_based_hashed, all_pos_triples_hashed, invert=True)
        mask = np.where(mask)[0]

        if mask.size == 0:
            raise Exception(
                "User selected filtered metric computation, but all corrupted triples exists"
                "also a positive triples.",
            )
        corrupted_object_based = corrupted_object_based[mask]

        return corrupted_subject_based, corrupted_object_based

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

    def _create_corrupted_triples(self, triple):
        candidate_entities_subject_based = self.all_entities[self.all_entities != triple[0:1]].reshape((-1, 1))
        candidate_entities_object_based = self.all_entities[self.all_entities != triple[2:3]].reshape((-1, 1))

        # Extract current test tuple: Either (subject,predicate) or (predicate,object)
        tuple_subject_based = np.reshape(a=triple[1:3], newshape=(1, 2))
        tuple_object_based = np.reshape(a=triple[0:2], newshape=(1, 2))

        # Copy current test tuple
        tuples_subject_based = np.repeat(
            a=tuple_subject_based,
            repeats=candidate_entities_subject_based.shape[0],
            axis=0,
        )
        tuples_object_based = np.repeat(
            a=tuple_object_based,
            repeats=candidate_entities_object_based.shape[0],
            axis=0,
        )

        corrupted_subject_based = np.concatenate(
            [
                candidate_entities_subject_based,
                tuples_subject_based,
            ],
            axis=1,
        )

        corrupted_object_based = np.concatenate(
            [
                tuples_object_based,
                candidate_entities_object_based,
            ],
            axis=1,
        )

        return corrupted_subject_based, corrupted_object_based

    def _compute_filtered_rank(
            self,
            model: BaseModule,
            pos_triple,
            corrupted_subject_based,
            corrupted_object_based,
            all_pos_triples_hashed,
    ) -> Tuple[int, int, float, float]:
        corrupted_subject_based, corrupted_object_based = self._filter_corrupted_triples(
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            all_pos_triples_hashed=all_pos_triples_hashed,
        )

        return self._compute_rank(
            model=model,
            pos_triple=pos_triple,
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            all_pos_triples_hashed=all_pos_triples_hashed,
        )

    def _compute_rank(
            self,
            model: BaseModule,
            pos_triple,
            corrupted_subject_based,
            corrupted_object_based,
            all_pos_triples_hashed=None,
    ) -> Tuple[int, int, float, float]:

        # Create tensors for numpy arrays
        corrupted_subject_based = torch.tensor(corrupted_subject_based, dtype=torch.long, device=self.device)
        corrupted_object_based = torch.tensor(corrupted_object_based, dtype=torch.long, device=self.device)

        scores_of_corrupted_subjects = model.predict_scores(corrupted_subject_based)
        scores_of_corrupted_objects = model.predict_scores(corrupted_object_based)

        score_of_positive = model.predict_scores(
            torch.tensor([pos_triple], dtype=torch.long, device=self.device),
        )

        rank_of_positive_subject_based, adj_rank_of_positive_subject_based = _compute_rank_from_scores(
            true_score=score_of_positive, all_scores=scores_of_corrupted_subjects,
        )
        rank_of_positive_object_based, adj_rank_of_positive_object_based = _compute_rank_from_scores(
            true_score=score_of_positive, all_scores=scores_of_corrupted_objects,
        )

        return (
            rank_of_positive_subject_based,
            rank_of_positive_object_based,
            adj_rank_of_positive_subject_based,
            adj_rank_of_positive_object_based,
        )

    def evaluate(self, test_triples: np.ndarray) -> MetricResults:
        start = timeit.default_timer()
        ranks: List[int] = []
        adj_ranks = np.empty(shape=(test_triples.shape[0], 2), dtype=np.float)
        hits_at_k_values = {
            k: []
            for k in self.hits_at_k
        }
        # Set eval mode in order to ignore functionalities such as dropout
        self.model = self.model.eval()

        all_pos_triples = np.concatenate([self.train_triples, test_triples], axis=0)
        all_pos_triples_hashed = np.apply_along_axis(self._hash_triples, 1, all_pos_triples)

        compute_rank_fct: Callable[..., Tuple[int, int, float, float]] = (
            self._compute_filtered_rank
            if self.filter_neg_triples else
            self._compute_rank
        )

        for i, pos_triple in enumerate(test_triples):
            corrupted_subject_based, corrupted_object_based = self._create_corrupted_triples(
                triple=pos_triple,
            )

            (
                rank_of_positive_subject_based,
                rank_of_positive_object_based,
                adjusted_rank_of_positive_subject_based,
                adjusted_rank_of_positive_object_based,
            ) = compute_rank_fct(
                kg_embedding_model=self.model,
                pos_triple=pos_triple,
                corrupted_subject_based=corrupted_subject_based,
                corrupted_object_based=corrupted_object_based,
                all_pos_triples_hashed=all_pos_triples_hashed,
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
