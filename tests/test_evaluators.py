# -*- coding: utf-8 -*-

"""Test the evaluators."""

import logging
import unittest
from typing import Any, ClassVar, Dict, Mapping, Optional, Type

import torch

from poem.evaluation import Evaluator, MetricResults, RankBasedEvaluator, RankBasedMetricResults
from poem.evaluation.evaluator import filter_scores_
from poem.evaluation.rank_based_evaluator import _compute_rank_from_scores

logger = logging.getLogger(__name__)


class _AbstractEvaluatorTests:
    """A test case for quickly defining common tests for evaluators models."""

    #: The evaluator to be tested
    evaluator_cls: ClassVar[Type[Evaluator]]
    evaluator_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    # Settings
    batch_size: int
    num_entities: int
    num_relations: int

    #: Test batch, scores, and mask
    batch: torch.LongTensor
    scores: torch.FloatTensor
    mask: torch.BoolTensor

    #: The evaluator instantiation
    evaluator: Evaluator

    def setUp(self) -> None:
        """Set up the test case."""
        self.batch_size = 8
        self.num_entities = 16
        self.num_relations = 4

        self.evaluator = self.evaluator_cls(**(self.evaluator_kwargs or {}))
        rand_subjects = torch.randint(self.num_entities, size=(self.batch_size,))
        rand_relations = torch.randint(self.num_relations, size=(self.batch_size,))
        rand_objects = torch.randint(self.num_entities, size=(self.batch_size,))
        self.batch = torch.stack([rand_subjects, rand_relations, rand_objects], dim=1)
        self.scores = torch.rand(self.batch_size, self.num_entities)

    def test_process_object_scores_(self) -> None:
        """Test the evaluator's ``process_object_scores_()`` function."""
        self.evaluator.process_object_scores_(
            batch=self.batch,
            scores=self.scores,
        )

    def test_process_subject_scores_(self) -> None:
        """Test the evaluator's ``process_subject_scores_()`` function."""
        self.evaluator.process_subject_scores_(
            batch=self.batch,
            scores=self.scores,
        )

    def test_finalize(self) -> None:
        # Process one batch
        self.evaluator.process_subject_scores_(
            batch=self.batch,
            scores=self.scores,
        )

        result = self.evaluator.finalize()
        assert isinstance(result, MetricResults)

        self._validate_result(
            result=result,
            data={'batch': self.batch, 'scores': self.scores}
        )

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        logger.warning(f'{self.__class__.__name__} did not overwrite _validate_result.')


class RankBasedEvaluatorTests(_AbstractEvaluatorTests, unittest.TestCase):
    """unittest for the RankBasedEvaluator."""

    evaluator_cls = RankBasedEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, RankBasedMetricResults)

        # Check value ranges
        assert 1 <= result.mean_rank <= self.num_entities
        assert 0 < result.mean_reciprocal_rank <= 1
        for k, v in result.hits_at_k.items():
            assert 0 <= v <= 1

        # TODO: Validate with data?


class EvaluatorUtilsTests(unittest.TestCase):
    """Test the utility functions used by evaluators."""

    def test_compute_rank_from_scores(self):
        """Test the _compute_rank_from_scores() function."""
        batch_size = 3
        all_scores = torch.tensor([
            [2., 2., 1., 3., 5.],
            [1., 1., 3., 4., 0.],
            [1., 1., 3., float('nan'), 0],
        ])
        # true_score: (2, 3, 3)
        true_score = torch.tensor([2., 3., 3.]).view(batch_size, 1)
        exp_avg_rank = torch.tensor([3.5, 2., 1., ])
        exp_adj_rank = exp_avg_rank / torch.tensor([(5 + 1) / 2, (5 + 1) / 2, (4 + 1) / 2])
        avg_rank, adj_rank = _compute_rank_from_scores(true_score=true_score, all_scores=all_scores)
        assert avg_rank.shape == (batch_size,)
        assert adj_rank.shape == (batch_size,)
        assert (avg_rank == exp_avg_rank).all(), (avg_rank, exp_avg_rank)
        assert (adj_rank == exp_adj_rank).all(), (adj_rank, exp_adj_rank)

    def test_filter_corrupted_triples(self):
        """Test the filter_corrupted_triples() function."""
        batch_size = 2
        num_entities = 4
        all_pos_triples = torch.tensor([
            [0, 1, 2],
            [1, 2, 3],
            [1, 3, 3],
            [3, 4, 1],
            [0, 2, 2],
            [3, 1, 2],
            [1, 2, 0],
        ], dtype=torch.long)
        batch = torch.tensor([
            [0, 1, 2],
            [1, 2, 3]
        ], dtype=torch.long)
        subject_filter_mask = torch.tensor([
            [True, False, False, False],
            [False, True, False, False],
        ], dtype=torch.bool)
        object_filter_mask = torch.tensor([
            [False, False, True, False],
            [False, False, False, True],
        ], dtype=torch.bool)
        exp_subject_filter_mask = torch.tensor([
            [True, False, False, True],
            [False, True, False, False],
        ], dtype=torch.bool)
        exp_object_filter_mask = torch.tensor([
            [False, False, True, False],
            [True, False, False, True],
        ], dtype=torch.bool)
        assert batch.shape == (batch_size, 3)
        assert subject_filter_mask.shape == (batch_size, num_entities)
        assert object_filter_mask.shape == (batch_size, num_entities)

        # Test subject scores
        subject_scores = torch.randn(batch_size, num_entities)
        old_subject_scores = subject_scores.detach().clone()
        filtered_subject_scores, relation_filter = filter_scores_(
            batch=batch,
            scores=subject_scores,
            all_pos_triples=all_pos_triples,
            filter_col=0,
        )
        # Assert in-place modification
        mask = torch.isfinite(subject_scores)
        assert (subject_scores[mask] == filtered_subject_scores[mask]).all()
        assert not torch.isfinite(filtered_subject_scores[~mask]).any()

        # Assert correct filtering
        assert (old_subject_scores[~exp_subject_filter_mask] == filtered_subject_scores[~exp_subject_filter_mask]).all()
        assert not torch.isfinite(filtered_subject_scores[exp_subject_filter_mask]).any()

        # Test object scores
        object_scores = torch.randn(batch_size, num_entities)
        old_object_scores = object_scores.detach().clone()
        filtered_object_scores, _ = filter_scores_(
            batch=batch,
            scores=object_scores,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=2,
        )
        # Assert in-place modification
        mask = torch.isfinite(object_scores)
        assert (object_scores[mask] == filtered_object_scores[mask]).all()
        assert not torch.isfinite(filtered_object_scores[~mask]).any()

        # Assert correct filtering
        assert (old_object_scores[~exp_object_filter_mask] == filtered_object_scores[~exp_object_filter_mask]).all()
        assert not torch.isfinite(filtered_object_scores[exp_object_filter_mask]).any()
