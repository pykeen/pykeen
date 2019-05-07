# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
import timeit
from dataclasses import dataclass
from typing import List, Iterable, Hashable, Callable, Tuple, Dict

import numpy as np
import torch

from poem.constants import FILTER_NEG_TRIPLES
from poem.evaluation.abstract_evaluator import AbstractEvalutor, EvaluatorConfig

log = logging.getLogger(__name__)


@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]


class RankBasedEvaluator(AbstractEvalutor):
    """."""

    def __init__(self, evaluator_config: EvaluatorConfig, hits_at_k=[1, 3, 5, 10]):
        super().__init__(evaluator_config=evaluator_config)
        self.all_entities = np.arange(0, len(self.entity_to_id))
        self.filter_neg_triples = self.evaluator_config.config.get(FILTER_NEG_TRIPLES)
        self.hits_at_k = hits_at_k
        self.train_triples = self.evaluator_config.training_triples

    def _hash_triples(self, triples: Iterable[Hashable]) -> int:
        """Hash a list of triples."""
        return hash(tuple(triples))

    def _update_hits_at_k(self,
                          hits_at_k_values: Dict[int, List[float]],
                          rank_of_positive_subject_based: int,
                          rank_of_positive_object_based: int
                          ) -> None:
        """Update the Hits@K dictionary for two values."""
        for k, values in hits_at_k_values.items():
            if rank_of_positive_subject_based < k:
                values.append(1.0)
            else:
                values.append(0.0)

            if rank_of_positive_object_based < k:
                values.append(1.0)
            else:
                values.append(0.0)

    def _create_corrupted_triples(self, triple):
        candidate_entities_subject_based = np.delete(arr=self.all_entities, obj=triple[0:1])
        candidate_entities_subject_based = np.reshape(candidate_entities_subject_based, newshape=(-1, 1))
        candidate_entities_object_based = np.delete(arr=self.all_entities, obj=triple[2:3])
        candidate_entities_object_based = np.reshape(candidate_entities_object_based, newshape=(-1, 1))

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
                      all_pos_triples_hashed=None,
                      ) -> Tuple[int, int]:
        """."""
        scores_of_corrupted_subjects = kg_embedding_model.predict(corrupted_subject_based)
        scores_of_corrupted_objects = kg_embedding_model.predict(corrupted_object_based)

        pos_triple = np.array(pos_triple)
        pos_triple = np.expand_dims(a=pos_triple, axis=0)
        pos_triple = torch.tensor(pos_triple, dtype=torch.long, device=self.device)

        score_of_positive = kg_embedding_model.predict(pos_triple)

        scores_subject_based = np.append(arr=scores_of_corrupted_subjects, values=score_of_positive)
        indice_of_pos_subject_based = scores_subject_based.size - 1

        scores_object_based = np.append(arr=scores_of_corrupted_objects, values=score_of_positive)
        indice_of_pos_object_based = scores_object_based.size - 1

        _, sorted_score_indices_subject_based = torch.sort(torch.tensor(scores_subject_based, dtype=torch.float),
                                                           descending=False)
        sorted_score_indices_subject_based = sorted_score_indices_subject_based.cpu().numpy()

        _, sorted_score_indices_object_based = torch.sort(torch.tensor(scores_object_based, dtype=torch.float),
                                                          descending=False)
        sorted_score_indices_object_based = sorted_score_indices_object_based.cpu().numpy()

        # Get index of first occurrence that fulfills the condition
        rank_of_positive_subject_based = np.where(sorted_score_indices_subject_based == indice_of_pos_subject_based)[0][
            0]
        rank_of_positive_object_based = np.where(sorted_score_indices_object_based == indice_of_pos_object_based)[0][0]

        return (
            rank_of_positive_subject_based,
            rank_of_positive_object_based,
        )

    def evaluate(self, test_triples):
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
