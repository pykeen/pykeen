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
from pykeen.evaluation.rank_based_evaluator import (
    RANK_EXPECTED_REALISTIC, RANK_OPTIMISTIC, RANK_PESSIMISTIC,
    RANK_REALISTIC, RANK_TYPES, SIDES, compute_rank_from_scores,
)
from pykeen.evaluation.sklearn import SklearnEvaluator, SklearnMetricResults
from pykeen.models import Model, TransE
from pykeen.triples import TriplesFactory
from pykeen.typing import MappedTriples
from tests.mocks import MockModel

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
            triples = self.factory.mapped_triples.to(self.model.device)
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
        result: RankBasedMetricResults

        # Check value ranges
        # check mean rank (MR)
        for side, all_type_mr in result.arithmetic_mean_rank.items():
            assert side in SIDES
            for rank_type, mr in all_type_mr.items():
                assert rank_type in RANK_TYPES
                assert isinstance(mr, float)
                assert 1 <= mr <= self.factory.num_entities

        # check mean reciprocal rank (MRR)
        for side, all_type_mrr in result.inverse_harmonic_mean_rank.items():
            assert side in SIDES
            for rank_type, mrr in all_type_mrr.items():
                assert rank_type in RANK_TYPES
                assert isinstance(mrr, float)
                assert 0 < mrr <= 1

        # check hits at k (H@k)
        for side, all_type_hits_at_k in result.hits_at_k.items():
            assert side in SIDES
            for rank_type, hits_at_k in all_type_hits_at_k.items():
                assert rank_type in RANK_TYPES
                for k, h in hits_at_k.items():
                    assert isinstance(k, int)
                    assert 0 < k < self.factory.num_entities
                    assert isinstance(h, float)
                    assert 0 <= h <= 1

        # check adjusted mean rank (AMR)
        for side, adjusted_mean_rank in result.adjusted_arithmetic_mean_rank.items():
            assert side in SIDES
            assert RANK_REALISTIC in adjusted_mean_rank
            assert isinstance(adjusted_mean_rank[RANK_REALISTIC], float)
            assert 0 < adjusted_mean_rank[RANK_REALISTIC] < 2

        # check adjusted mean rank index (AMRI)
        for side, adjusted_mean_rank_index in result.adjusted_arithmetic_mean_rank_index.items():
            assert side in SIDES
            assert RANK_REALISTIC in adjusted_mean_rank_index
            assert isinstance(adjusted_mean_rank_index[RANK_REALISTIC], float)
            assert -1 <= adjusted_mean_rank_index[RANK_REALISTIC] <= 1

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
        scores = data['scores'].detach().cpu().numpy()
        mask = data['mask'].detach().cpu().float().numpy()

        # filtering
        uniq = dict()
        batch = data['batch'].detach().cpu().numpy()
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
        true_score = torch.as_tensor([2., 3., 3.]).view(batch_size, 1)
        exp_best_rank = torch.as_tensor([3., 2., 1.])
        exp_worst_rank = torch.as_tensor([4., 2., 1.])
        exp_avg_rank = 0.5 * (exp_best_rank + exp_worst_rank)
        exp_exp_rank = torch.as_tensor([(5 + 1) / 2, (5 + 1) / 2, (4 + 1) / 2])
        ranks = compute_rank_from_scores(true_score=true_score, all_scores=all_scores)

        optimistic_rank = ranks.get(RANK_OPTIMISTIC)
        assert optimistic_rank.shape == (batch_size,)
        assert (optimistic_rank == exp_best_rank).all()

        pessimistic_rank = ranks.get(RANK_PESSIMISTIC)
        assert pessimistic_rank.shape == (batch_size,)
        assert (pessimistic_rank == exp_worst_rank).all()

        realistic_rank = ranks.get(RANK_REALISTIC)
        assert realistic_rank.shape == (batch_size,)
        assert (realistic_rank == exp_avg_rank).all(), (realistic_rank, exp_avg_rank)

        expected_realistic_rank = ranks.get(RANK_EXPECTED_REALISTIC)
        assert expected_realistic_rank is not None
        assert expected_realistic_rank.shape == (batch_size,)
        assert (expected_realistic_rank == exp_exp_rank).all(), (expected_realistic_rank, exp_exp_rank)

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

    def __init__(self, *, counter: int, filtered: bool, automatic_memory_optimization: bool = True) -> None:
        super().__init__(filtered=filtered, automatic_memory_optimization=automatic_memory_optimization)
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
            arithmetic_mean_rank=self.counter,
            geometric_mean_rank=None,
            harmonic_mean_rank=None,
            median_rank=None,
            inverse_arithmetic_mean_rank=None,
            inverse_geometric_mean_rank=None,
            inverse_harmonic_mean_rank=None,
            inverse_median_rank=None,
            rank_std=None,
            rank_var=None,
            rank_mad=None,
            adjusted_arithmetic_mean_rank=None,
            adjusted_arithmetic_mean_rank_index=None,
            hits_at_k=dict(),
        )

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(losses={self.losses})'


class TestEvaluationStructure(unittest.TestCase):
    """Tests for testing the correct structure of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation structure."""
        self.counter = 1337
        self.evaluator = DummyEvaluator(counter=self.counter, filtered=True, automatic_memory_optimization=False)
        self.dataset = Nations()
        self.model = MockModel(triples_factory=self.dataset.training)

    def test_evaluation_structure(self):
        """Test if the evaluator has a balanced call of head and tail processors."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            additional_filter_triples=self.dataset.training.mapped_triples,
            mapped_triples=self.dataset.testing.mapped_triples,
            batch_size=1,
            use_tqdm=False,
        )
        self.assertIsInstance(eval_results, RankBasedMetricResults)
        assert eval_results.arithmetic_mean_rank == self.counter, 'Should end at the same value as it started'


class TestEvaluationFiltering(unittest.TestCase):
    """Tests for testing the correct filtering of positive triples of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation filtering."""
        self.evaluator = RankBasedEvaluator(filtered=True, automatic_memory_optimization=False)
        self.triples_factory = Nations().training
        self.model = MockModel(triples_factory=self.triples_factory)

        # The MockModel gives the highest score to the highest entity id
        max_score = self.triples_factory.num_entities - 1

        # The test triples are created to yield the third highest score on both head and tail prediction
        self.test_triples = torch.tensor([[max_score - 2, 0, max_score - 2]])

        # Write new mapped triples to the model, since the model's triples will be used to filter
        # These triples are created to yield the highest score on both head and tail prediction for the
        # test triple at hand
        self.training_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score],
                [max_score, 0, max_score - 2],
            ],
        )

        # The validation triples are created to yield the second highest score on both head and tail prediction for the
        # test triple at hand
        self.validation_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score - 1],
                [max_score - 1, 0, max_score - 2],
            ],
        )

    def test_evaluation_filtering_without_validation_triples(self):
        """Test if the evaluator's triple filtering works as expected."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.test_triples,
            additional_filter_triples=self.training_triples,
            batch_size=1,
            use_tqdm=False,
        )
        assert eval_results.arithmetic_mean_rank['both']['realistic'] == 2, 'The rank should equal 2'

    def test_evaluation_filtering_with_validation_triples(self):
        """Test if the evaluator's triple filtering works as expected when including additional filter triples."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.test_triples,
            additional_filter_triples=[
                self.training_triples,
                self.validation_triples,
            ],
            batch_size=1,
            use_tqdm=False,
        )
        assert eval_results.arithmetic_mean_rank['both']['realistic'] == 1, 'The rank should equal 1'
