# -*- coding: utf-8 -*-

"""Test the evaluators."""

import dataclasses
import logging
import unittest
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple, Type

import torch

from pykeen.datasets import Nations
from pykeen.evaluation import Evaluator, MetricResults, RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.evaluator import create_dense_positive_mask_, create_sparse_positive_filter_, filter_scores_
from pykeen.evaluation.rank_based_evaluator import RANK_TYPES, SIDES, compute_rank_from_scores
from pykeen.evaluation.sklearn import SklearnEvaluator, SklearnMetricResults
from pykeen.models import TransE
from pykeen.models.base import EntityRelationEmbeddingModel, Model
from pykeen.triples import TriplesFactory
from pykeen.typing import MappedTriples

logger = logging.getLogger(__name__)


class _AbstractEvaluatorTests:
    """A test case for quickly defining common tests for evaluators models."""

    # The triples factory and model
    factory: TriplesFactory
    model: Model

    #: The evaluator to be tested
    evaluator_cls: ClassVar[Type[Evaluator]]
    evaluator_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    # Settings
    batch_size: int
    embedding_dim: int

    #: The evaluator instantiation
    evaluator: Evaluator

    def setUp(self) -> None:
        """Set up the test case."""
        # Settings
        self.batch_size = 8
        self.embedding_dim = 7

        # Initialize evaluator
        self.evaluator = self.evaluator_cls(**(self.evaluator_kwargs or {}))

        # Use small test dataset
        self.factory = Nations().training

        # Use small model (untrained)
        self.model = TransE(triples_factory=self.factory, embedding_dim=self.embedding_dim)

    def _get_input(
        self,
        inverse: bool = False,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Optional[torch.BoolTensor]]:
        # Get batch
        hrt_batch = self.factory.mapped_triples[:self.batch_size].to(self.model.device)

        # Compute scores
        if inverse:
            scores = self.model.score_h(rt_batch=hrt_batch[:, 1:])
        else:
            scores = self.model.score_t(hr_batch=hrt_batch[:, :2])

        # Compute mask only if required
        if self.evaluator.requires_positive_mask:
            # TODO: Re-use filtering code
            triples = self.factory.mapped_triples
            if inverse:
                sel_col, start_col = 0, 1
            else:
                sel_col, start_col = 2, 0
            stop_col = start_col + 2

            # shape: (batch_size, num_triples)
            triple_mask = (triples[None, :, start_col:stop_col] == hrt_batch[:, None, start_col:stop_col]).all(dim=-1)
            batch_indices, triple_indices = triple_mask.nonzero(as_tuple=True)
            entity_indices = triples[triple_indices, sel_col]

            # shape: (batch_size, num_entities)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[batch_indices, entity_indices] = True
        else:
            mask = None

        return hrt_batch, scores, mask

    def test_process_tail_scores_(self) -> None:
        """Test the evaluator's ``process_tail_scores_()`` function."""
        hrt_batch, scores, mask = self._get_input()
        true_scores = scores[torch.arange(0, hrt_batch.shape[0]), hrt_batch[:, 2]][:, None]
        self.evaluator.process_tail_scores_(
            hrt_batch=hrt_batch,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

    def test_process_head_scores_(self) -> None:
        """Test the evaluator's ``process_head_scores_()`` function."""
        hrt_batch, scores, mask = self._get_input(inverse=True)
        true_scores = scores[torch.arange(0, hrt_batch.shape[0]), hrt_batch[:, 0]][:, None]
        self.evaluator.process_head_scores_(
            hrt_batch=hrt_batch,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

    def test_finalize(self) -> None:
        # Process one batch
        hrt_batch, scores, mask = self._get_input()
        true_scores = scores[torch.arange(0, hrt_batch.shape[0]), hrt_batch[:, 2]][:, None]
        self.evaluator.process_tail_scores_(
            hrt_batch=hrt_batch,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

        result = self.evaluator.finalize()
        assert isinstance(result, MetricResults)

        self._validate_result(
            result=result,
            data={'batch': hrt_batch, 'scores': scores, 'mask': mask},
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
        for side, all_type_mr in result.mean_rank.items():
            assert side in SIDES
            for rank_type, mr in all_type_mr.items():
                assert rank_type in RANK_TYPES
                assert isinstance(mr, float)
                assert 1 <= mr <= self.factory.num_entities
        for side, all_type_mrr in result.mean_reciprocal_rank.items():
            assert side in SIDES
            for rank_type, mrr in all_type_mrr.items():
                assert rank_type in RANK_TYPES
                assert isinstance(mrr, float)
                assert 0 < mrr <= 1
        for side, all_type_hits_at_k in result.hits_at_k.items():
            assert side in SIDES
            for rank_type, hits_at_k in all_type_hits_at_k.items():
                assert rank_type in RANK_TYPES
                for k, h in hits_at_k.items():
                    assert isinstance(k, int)
                    assert 0 < k < self.factory.num_entities
                    assert isinstance(h, float)
                    assert 0 <= h <= 1

        # TODO: Validate with data?


class SklearnEvaluatorTest(_AbstractEvaluatorTests, unittest.TestCase):
    """Unittest for the SklearnEvaluator."""

    evaluator_cls = SklearnEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, SklearnMetricResults)

        # check value
        scores = data['scores'].detach().numpy()
        mask = data['mask'].detach().float().numpy()

        # filtering
        uniq = dict()
        batch = data['batch'].detach().numpy()
        for i, (h, r) in enumerate(batch[:, :2]):
            uniq[int(h), int(r)] = i
        indices = sorted(uniq.values())
        mask = mask[indices]
        scores = scores[indices]

        for field in dataclasses.fields(SklearnMetricResults):
            f = field.metadata['f']
            exp_score = f(mask.flat, scores.flat)
            self.assertAlmostEqual(result.get_metric(field.name), exp_score)


class EvaluatorUtilsTests(unittest.TestCase):
    """Test the utility functions used by evaluators."""

    def setUp(self) -> None:
        """Set up the test case with a fixed random seed."""
        self.generator = torch.random.manual_seed(seed=42)

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
        exp_best_rank = torch.tensor([3., 2., 1.])
        exp_worst_rank = torch.tensor([4., 2., 1.])
        exp_avg_rank = 0.5 * (exp_best_rank + exp_worst_rank)
        exp_adj_rank = exp_avg_rank / torch.tensor([(5 + 1) / 2, (5 + 1) / 2, (4 + 1) / 2])
        ranks = compute_rank_from_scores(true_score=true_score, all_scores=all_scores)

        best_rank = ranks.get('best')
        assert best_rank.shape == (batch_size,)
        assert (best_rank == exp_best_rank).all()

        worst_rank = ranks.get('worst')
        assert worst_rank.shape == (batch_size,)
        assert (worst_rank == exp_worst_rank).all()

        avg_rank = ranks.get('avg')
        assert avg_rank.shape == (batch_size,)
        assert (avg_rank == exp_avg_rank).all(), (avg_rank, exp_avg_rank)

        adj_rank = ranks.get('adj')
        assert adj_rank.shape == (batch_size,)
        assert (adj_rank == exp_adj_rank).all(), (adj_rank, exp_adj_rank)

    def test_create_sparse_positive_filter_(self):
        """Test method create_sparse_positive_filter_."""
        batch_size = 4
        factory = Nations().training
        all_triples = factory.mapped_triples
        batch = all_triples[:batch_size, :]

        # head based filter
        sparse_positives, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_triples,
            relation_filter=None,
            filter_col=0,
        )

        # preprocessing for faster lookup
        triples = set()
        for trip in all_triples.detach().numpy():
            triples.add(tuple(map(int, trip)))

        # check that all found positives are positive
        for batch_id, entity_id in sparse_positives:
            same = batch[batch_id, 1:]
            assert (int(entity_id),) + tuple(map(int, same)) in triples

    def test_create_dense_positive_mask_(self):
        """Test method create_dense_positive_mask_."""
        batch_size = 3
        num_positives = 5
        num_entities = 7
        zero_tensor = torch.zeros(batch_size, num_entities)
        filter_batch = torch.empty(num_positives, 2, dtype=torch.long)
        for i in range(num_positives):
            filter_batch[i, 0] = i % batch_size
        filter_batch[:, 1] = torch.randperm(num_positives, generator=self.generator)
        dense_mask = create_dense_positive_mask_(zero_tensor=zero_tensor, filter_batch=filter_batch)

        # check in-place
        assert id(dense_mask) == id(zero_tensor)

        for b in range(batch_size):
            for e in range(num_entities):
                if (torch.as_tensor([b, e]).view(1, 2) == filter_batch).all(dim=1).any():
                    assert dense_mask[b, e] == 1
                else:
                    assert dense_mask[b, e] == 0

    def test_filter_corrupted_triples(self):
        """Test the filter_corrupted_triples() function."""
        batch_size = 2
        num_entities = 4
        all_pos_triples = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [1, 3, 3],
                [3, 4, 1],
                [0, 2, 2],
                [3, 1, 2],
                [1, 2, 0],
            ], dtype=torch.long,
        )
        batch = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ], dtype=torch.long,
        )
        head_filter_mask = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
            ], dtype=torch.bool,
        )
        tail_filter_mask = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, True],
            ], dtype=torch.bool,
        )
        exp_head_filter_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
            ], dtype=torch.bool,
        )
        exp_tail_filter_mask = torch.tensor(
            [
                [False, False, True, False],
                [True, False, False, True],
            ], dtype=torch.bool,
        )
        assert batch.shape == (batch_size, 3)
        assert head_filter_mask.shape == (batch_size, num_entities)
        assert tail_filter_mask.shape == (batch_size, num_entities)

        # Test head scores
        head_scores = torch.randn(batch_size, num_entities, generator=self.generator)
        old_head_scores = head_scores.detach().clone()
        positive_filter_heads, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=None,
            filter_col=0,
        )
        filtered_head_scores = filter_scores_(
            scores=head_scores,
            filter_batch=positive_filter_heads,
        )
        # Assert in-place modification
        mask = torch.isfinite(head_scores)
        assert (head_scores[mask] == filtered_head_scores[mask]).all()
        assert not torch.isfinite(filtered_head_scores[~mask]).any()

        # Assert correct filtering
        assert (old_head_scores[~exp_head_filter_mask] == filtered_head_scores[~exp_head_filter_mask]).all()
        assert not torch.isfinite(filtered_head_scores[exp_head_filter_mask]).any()

        # Test tail scores
        tail_scores = torch.randn(batch_size, num_entities, generator=self.generator)
        old_tail_scores = tail_scores.detach().clone()
        positive_filter_tails, _ = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=2,
        )
        filtered_tail_scores = filter_scores_(
            scores=tail_scores,
            filter_batch=positive_filter_tails,
        )
        # Assert in-place modification
        mask = torch.isfinite(tail_scores)
        assert (tail_scores[mask] == filtered_tail_scores[mask]).all()
        assert not torch.isfinite(filtered_tail_scores[~mask]).any()

        # Assert correct filtering
        assert (old_tail_scores[~exp_tail_filter_mask] == filtered_tail_scores[~exp_tail_filter_mask]).all()
        assert not torch.isfinite(filtered_tail_scores[exp_tail_filter_mask]).any()


class DummyEvaluator(Evaluator):
    """A dummy evaluator for testing the structure of the evaluation function."""

    def __init__(self, *, counter: int, filtered: bool) -> None:
        super().__init__(filtered=filtered)
        self.counter = counter

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self.counter += 1

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self.counter -= 1

    def finalize(self) -> MetricResults:  # noqa: D102
        return RankBasedMetricResults(
            mean_rank=self.counter,
            mean_reciprocal_rank=None,
            adjusted_mean_rank=None,
            hits_at_k=dict(),
        )

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(losses={self.losses})'


class DummyModel(EntityRelationEmbeddingModel):
    """A dummy model returning fake scores."""

    def __init__(self, triples_factory: TriplesFactory, automatic_memory_optimization: bool):
        super().__init__(triples_factory=triples_factory, automatic_memory_optimization=automatic_memory_optimization)
        num_entities = self.num_entities
        self.scores = torch.arange(num_entities, dtype=torch.float)

    def _generate_fake_scores(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Generate fake scores s[b, i] = i of size (batch_size, num_entities)."""
        batch_size = batch.shape[0]
        batch_scores = self.scores.view(1, -1).repeat(batch_size, 1)
        assert batch_scores.shape == (batch_size, self.num_entities)
        return batch_scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=hrt_batch)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=hr_batch)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=rt_batch)

    def reset_parameters_(self) -> Model:  # noqa: D102
        pass  # Not needed for unittest


class TestEvaluationStructure(unittest.TestCase):
    """Tests for testing the correct structure of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation structure."""
        self.counter = 1337
        self.evaluator = DummyEvaluator(counter=self.counter, filtered=True)
        self.triples_factory = Nations().training
        self.model = DummyModel(triples_factory=self.triples_factory, automatic_memory_optimization=False)

    def test_evaluation_structure(self):
        """Test if the evaluator has a balanced call of head and tail processors."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.triples_factory.mapped_triples,
            batch_size=1,
        )
        assert eval_results.mean_rank == self.counter, 'Should end at the same value as it started'
