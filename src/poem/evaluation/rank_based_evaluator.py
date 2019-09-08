# -*- coding: utf-8 -*-

"""Implementation of ranked based evaluator."""

import logging
import timeit
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .evaluator import Evaluator, MetricResults
from ..models.base import BaseModule
from ..training.utils import split_list_in_batches
from ..typing import MappedTriples

__all__ = [
    'RankBasedEvaluator',
]

logger = logging.getLogger(__name__)


def _compute_rank_from_scores(true_score, all_scores) -> Tuple[int, float]:
    """Compute rank and adjusted rank given scores.

    :param true_score: torch.Tensor, shape: (batch_size, 1)
        The score of the true triple.
    :param all_scores: torch.Tensor, shape: (batch_size, num_entities)
        The scores of all corrupted triples.
    :return: a tuple (best_rank, adjusted_avg_rank) where
        best_rank: np.ndarray (batch_size, 1)
            The rank of the true triple as given as the number of elements having a better score plus one.
        adjusted_avg_rank: np.ndarray (batch_size, 1)
            The avg rank of the true triple divided by the expected rank in random scoring.
    """
    best_rank = (all_scores > true_score).sum(dim=1) + 1
    worst_rank = (all_scores >= true_score).sum(dim=1) + 1
    avg_rank = (best_rank + worst_rank).to(dtype=torch.float) * 0.5
    adjusted_avg_rank = avg_rank / ((all_scores.shape[1] + 1) * 0.5)
    return best_rank.detach().cpu().numpy(), adjusted_avg_rank.detach().cpu().numpy()


class RankBasedEvaluator(Evaluator):
    """A rank-based evaluator for KGE models."""

    def __init__(
            self,
            model: BaseModule = None,
            filter_neg_triples: bool = False,
            hits_at_k: Optional[List[int]] = None,
    ) -> None:
        """Initialize the evaluator.

        :param model
            The fitted KGE model that is to be evaluated.
        :param filter_neg_triples
            Indicating whether negative triples should be filtered, by default 'False'
        :param hits_at_k
            A list containing all integers that represent the 'k's to be used for the hits@k evaluation,
            by default [1, 3, 5, 10]
        """
        super().__init__(model=model)
        self.filter_neg_triples = filter_neg_triples
        self.hits_at_k = hits_at_k if hits_at_k is not None else [1, 3, 5, 10]

    def _filter_corrupted_triples(
            self,
            batch,
            subject_batch,
            object_batch,
            all_pos_triples,
    ):
        # TODO: Make static method / function
        subjects = batch[:, 0:1]
        relations = batch[:, 1:2]
        objects = batch[:, 2:3]
        batch_size = batch.shape[0]

        subject_filter = (all_pos_triples[:, 0:1]).view(1, -1).repeat(batch_size, 1) == subjects
        relation_filter = (all_pos_triples[:, 1:2]).view(1, -1).repeat(batch_size, 1) == relations
        object_filter = (all_pos_triples[:, 2:3]).view(1, -1).repeat(batch_size, 1) == objects

        # Short objects batch list
        pairs_filter = (subject_filter & relation_filter)
        objects_in_triples = (all_pos_triples[:, 2:3]).view(1, -1).repeat(batch_size, 1)[pairs_filter]
        row_indices = pairs_filter.nonzero()[:, 0]
        object_batch[row_indices, objects_in_triples] = 1

        # Short subjects batch list
        pairs_filter = (object_filter & relation_filter)
        subjects_in_triples = (all_pos_triples[:, 0:1]).view(1, -1).repeat(batch_size, 1)[pairs_filter]
        row_indices = pairs_filter.nonzero()[:, 0]
        subject_batch[row_indices, subjects_in_triples] = 1

        # TODO: Create warning when all triples will be filtered
        # if mask.size == 0:
        #     raise Exception("User selected filtered metric computation, but all corrupted triples exists"
        #                     "also as positive triples.")

        return subject_batch, object_batch

    def _update_hits_at_k(
            self,
            hits_at_k_values: Dict[int, List[float]],
            ranks: List[int],
    ) -> None:
        """Update the Hits@K dictionary for two values."""
        # TODO: Make static method / function
        for k, values in hits_at_k_values.items():
            hits_at_k = (np.array(ranks) <= k) * 1
            values.extend(hits_at_k)

    def _compute_filtered_rank(
            self,
            batch,
            subject_batch,
            object_batch,
            all_pos_triples,
    ) -> Tuple[int, int, float, float]:
        subject_batch, object_batch = self._filter_corrupted_triples(
            batch=batch,
            subject_batch=subject_batch,
            object_batch=object_batch,
            all_pos_triples=all_pos_triples,
        )

        return self._compute_rank(
            batch=batch,
            subject_batch=subject_batch,
            object_batch=object_batch,
            all_pos_triples=all_pos_triples,
        )

    def _compute_rank(
            self,
            batch,
            subject_batch,
            object_batch,
            all_pos_triples,
    ) -> Tuple[int, int, float, float]:
        subjects = batch[:, 0:1]
        objects = batch[:, 2:3]
        batch_size = batch.shape[0]

        scores_of_corrupted_subjects_batch = self.model.predict_scores_all_subjects(batch[:, 1:3])
        score_of_positive_subject_batch = (
            scores_of_corrupted_subjects_batch[torch.arange(0, batch_size), subjects.flatten()]
        ).view(-1, 1)
        scores_of_corrupted_subjects_batch[subject_batch] = score_of_positive_subject_batch.min() - 1

        scores_of_corrupted_objects_batch = self.model.predict_scores_all_objects(batch[:, 0:2])
        score_of_positive_object_batch = (
            scores_of_corrupted_objects_batch[torch.arange(0, batch_size), objects.flatten()]
        ).view(-1, 1)
        scores_of_corrupted_objects_batch[object_batch] = score_of_positive_object_batch.min() - 1

        rank_of_positive_subject_based, adj_rank_of_positive_subject_based = _compute_rank_from_scores(
            true_score=score_of_positive_subject_batch, all_scores=scores_of_corrupted_subjects_batch,
        )
        rank_of_positive_object_based, adj_rank_of_positive_object_based = _compute_rank_from_scores(
            true_score=score_of_positive_object_batch, all_scores=scores_of_corrupted_objects_batch,
        )

        return (
            rank_of_positive_subject_based,
            rank_of_positive_object_based,
            adj_rank_of_positive_subject_based,
            adj_rank_of_positive_object_based,
        )

    def evaluate(self, mapped_triples: MappedTriples, batch_size: int = 1) -> MetricResults:
        """Evaluate a given KGE model based on a test-set of mapped triples.

        :param mapped_triples: np.ndarray, shape: (number of triples, 3)
            The mapped triples to be used for evaluation.
        :param batch_size: int, optional
            The batch size to be used for the evaluation. A bigger batch size increases the utilization of GPUs during
            evaluation. However due to memory limitations, the maximum possible batch size depends on the number of
            entities contained in the triples when using the setting without filtering negative triples and the amount
            of triples in the train and test set for the filtered setting. By default '1'
        :return: MetricResults
            The dataclass containing the mean_rank, mean_reciprocal_rank, adjusted_mean_rank,
            adjusted_mean_reciprocal_rank and hits_at_k results from the evaluation.
        """
        start = timeit.default_timer()
        ranks: List[int] = []
        adj_ranks: List[int] = []
        hits_at_k_values = {
            k: []
            for k in self.hits_at_k
        }
        # Set eval mode in order to ignore functionalities such as dropout
        self.model = self.model.eval()

        all_pos_triples = torch.cat([self.model.triples_factory.mapped_triples, mapped_triples], dim=0)
        all_pos_triples = all_pos_triples.to(device=self.device)
        all_entities = self.model.triples_factory.all_entities.to(device=self.device)

        mapped_triples = mapped_triples.to(device=self.device)

        compute_rank_fct: Callable[..., Tuple[int, int, float, float]] = (
            self._compute_filtered_rank
            if self.filter_neg_triples else
            self._compute_rank
        )

        batches = split_list_in_batches(input_list=mapped_triples, batch_size=batch_size)

        num_triples = mapped_triples.shape[0]

        with tqdm(
                desc=f'⚡️ Evaluating triples ',
                total=num_triples,
                unit='triple(s)',
                unit_scale=True,
        ) as progress_bar:
            for i, batch in enumerate(batches):
                subjects = batch[:, 0:1]
                objects = batch[:, 2:3]
                batch_size = subjects.shape[0]

                subject_batch = all_entities.repeat(batch_size, 1) == subjects
                object_batch = all_entities.repeat(batch_size, 1) == objects

                # Disable gradient tracking
                with torch.no_grad():
                    (
                        rank_of_positive_subject_based,
                        rank_of_positive_object_based,
                        adjusted_rank_of_positive_subject_based,
                        adjusted_rank_of_positive_object_based,
                    ) = compute_rank_fct(
                        batch=batch,
                        subject_batch=subject_batch,
                        object_batch=object_batch,
                        all_pos_triples=all_pos_triples,
                    )

                ranks.extend(rank_of_positive_subject_based)
                ranks.extend(rank_of_positive_object_based)
                adj_ranks.extend(adjusted_rank_of_positive_subject_based)
                adj_ranks.extend(adjusted_rank_of_positive_object_based)

                progress_bar.update(batch_size)

        # Compute hits@k for k in {1,3,5,10}
        self._update_hits_at_k(
            hits_at_k_values,
            ranks,
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
        logger.info("Evaluation took %.2fs seconds", stop - start)

        return MetricResults(
            mean_rank=mean_rank,
            mean_reciprocal_rank=mean_reciprocal_rank,
            hits_at_k=hits_at_k,
            adjusted_mean_rank=adjusted_mean_rank,
            adjusted_mean_reciprocal_rank=adjusted_mean_reciprocal_rank,
        )
