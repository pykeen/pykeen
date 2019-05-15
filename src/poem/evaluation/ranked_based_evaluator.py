# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
import timeit
from dataclasses import dataclass
from typing import List, Iterable, Hashable, Callable, Tuple, Dict

import numpy as np
import torch

from poem.constants import Complex_LITERAL_NAME_CWA, COMPLEX_CWA_NAME, DISTMULT_LITERAL_NAME_CWA, \
    DISTMULT_LITERAL_NAME_OWA, TRANS_E_NAME
from poem.evaluation.abstract_evaluator import AbstractEvalutor

log = logging.getLogger(__name__)


@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]


class RankBasedEvaluator(AbstractEvalutor):
    """."""

    def __init__(self, kge_model, entity_to_id, relation_to_id, training_triples: np.ndarray, filter_neg_triples=False,
                 hits_at_k=[1, 3, 5, 10]):
        super().__init__(kge_model=kge_model, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
        self.all_entities = np.arange(0, len(self.entity_to_id))
        self.filter_neg_triples = filter_neg_triples
        self.hits_at_k = hits_at_k
        self.train_triples = training_triples
        self.kge_to_descend_sorting = {
            COMPLEX_CWA_NAME: True,
            Complex_LITERAL_NAME_CWA: True,
            DISTMULT_LITERAL_NAME_CWA: True,
            DISTMULT_LITERAL_NAME_OWA: True,
            TRANS_E_NAME: False

        }

    def _hash_triples(self, triples: Iterable[Hashable]) -> int:
        """Hash a list of triples."""
        return hash(tuple(triples))

    def _filter_corrupted_triples(self,
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
            raise Exception("User selected filtered metric computation, but all corrupted triples exists"
                            "also a positive triples.")
        corrupted_object_based = corrupted_object_based[mask]

        return corrupted_subject_based, corrupted_object_based

    def _update_hits_at_k(self,
                          hits_at_k_values: Dict[int, List[float]],
                          rank_of_positive_subject_based: int,
                          rank_of_positive_object_based: int
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
        tuples_subject_based = np.repeat(a=tuple_subject_based, repeats=candidate_entities_subject_based.shape[0],
                                         axis=0)
        tuples_object_based = np.repeat(a=tuple_object_based, repeats=candidate_entities_object_based.shape[0], axis=0)

        corrupted_subject_based = np.concatenate([candidate_entities_subject_based, tuples_subject_based], axis=1)
        corrupted_subject_based = torch.tensor(corrupted_subject_based, dtype=torch.long, device=self.device)

        corrupted_object_based = np.concatenate([tuples_object_based, candidate_entities_object_based], axis=1)
        corrupted_object_based = torch.tensor(corrupted_object_based, dtype=torch.long, device=self.device)

        return corrupted_subject_based, corrupted_object_based

    def _compute_filtered_rank(self,
                               kg_embedding_model,
                               pos_triple,
                               corrupted_subject_based,
                               corrupted_object_based,
                               all_pos_triples_hashed,
                               ) -> Tuple[int, int]:
        """."""
        corrupted_subject_based, corrupted_object_based = self._filter_corrupted_triples(
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            all_pos_triples_hashed=all_pos_triples_hashed)

        return self._compute_rank(
            kg_embedding_model=kg_embedding_model,
            pos_triple=pos_triple,
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            device=self.device,
            all_pos_triples_hashed=all_pos_triples_hashed,
        )

    def _compute_rank(self,
                      kg_embedding_model,
                      pos_triple,
                      corrupted_subject_based,
                      corrupted_object_based,
                      all_pos_triples_hashed=None
                      ) -> Tuple[int, int]:
        """."""
        scores_of_corrupted_subjects = kg_embedding_model.predict(corrupted_subject_based)
        scores_of_corrupted_objects = kg_embedding_model.predict(corrupted_object_based)

        score_of_positive = kg_embedding_model.predict(torch.tensor([pos_triple], dtype=torch.long, device=self.device))

        rank_of_positive_subject_based = scores_of_corrupted_subjects.shape[0] - \
                                         np.greater_equal(scores_of_corrupted_subjects, score_of_positive).sum()

        rank_of_positive_object_based = scores_of_corrupted_objects.shape[0] - \
                                        np.greater_equal(scores_of_corrupted_objects, score_of_positive).sum()

        return (
            rank_of_positive_subject_based + 1,
            rank_of_positive_object_based + 1,
        )

    def evaluate(self, test_triples: np.ndarray):
        """."""
        start = timeit.default_timer()
        ranks: List[int] = []
        hits_at_k_values = {
            k: []
            for k in self.hits_at_k
        }
        # Set eval mode in order to ignore functionalities such as dropout
        self.kge_model = self.kge_model.eval()

        all_pos_triples = np.concatenate([self.train_triples, test_triples], axis=0)
        all_pos_triples_hashed = np.apply_along_axis(self._hash_triples, 1, all_pos_triples)

        compute_rank_fct: Callable[..., Tuple[int, int]] = (
            self._compute_filtered_rank
            if self.filter_neg_triples else
            self._compute_rank
        )

        for pos_triple in test_triples:
            corrupted_subject_based, corrupted_object_based = self._create_corrupted_triples(
                triple=pos_triple)

            rank_of_positive_subject_based, rank_of_positive_object_based = compute_rank_fct(
                kg_embedding_model=self.kge_model,
                pos_triple=pos_triple,
                corrupted_subject_based=corrupted_subject_based,
                corrupted_object_based=corrupted_object_based,
                all_pos_triples_hashed=all_pos_triples_hashed,
            )

            ranks.append(rank_of_positive_subject_based)
            ranks.append(rank_of_positive_object_based)

            # Compute hits@k for k in {1,3,5,10}
            self._update_hits_at_k(
                hits_at_k_values,
                rank_of_positive_subject_based=rank_of_positive_subject_based,
                rank_of_positive_object_based=rank_of_positive_object_based,
            )

        mean_rank = float(np.mean(ranks))
        hits_at_k: Dict[int, float] = {
            k: np.mean(values)
            for k, values in hits_at_k_values.items()
        }

        stop = timeit.default_timer()
        log.info("Evaluation took %.2fs seconds", stop - start)

        return MetricResults(
            mean_rank=mean_rank,
            hits_at_k=hits_at_k,
        )
