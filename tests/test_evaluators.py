# -*- coding: utf-8 -*-

"""Test the evaluators."""

import logging
import unittest
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple, Type

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from poem.datasets import NationsTrainingTriplesFactory
from poem.evaluation import Evaluator, MetricResults, RankBasedEvaluator, RankBasedMetricResults
from poem.evaluation.evaluator import create_dense_positive_mask_, create_sparse_positive_filter_, filter_scores_
from poem.evaluation.rank_based_evaluator import compute_rank_from_scores
from poem.evaluation.sklearn import SklearnEvaluator, SklearnMetricResults
from poem.models import BaseModule, TransE
from poem.triples import TriplesFactory
from poem.typing import MappedTriples

logger = logging.getLogger(__name__)


class _AbstractEvaluatorTests:
    """A test case for quickly defining common tests for evaluators models."""

    # The triples factory and model
    factory: TriplesFactory
    model: BaseModule

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
        self.factory = NationsTrainingTriplesFactory()

        # Use small model (untrained)
        self.model = TransE(triples_factory=self.factory, embedding_dim=self.embedding_dim)

    def _get_input(
        self,
        inverse: bool = False,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Optional[torch.BoolTensor]]:
        # Get batch
        batch = self.factory.mapped_triples[:self.batch_size]

        # Compute scores
        if inverse:
            scores = self.model.forward_inverse_cwa(batch=batch[:, 1:])
        else:
            scores = self.model.forward_cwa(batch=batch[:, :2])

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
            triple_mask = (triples[None, :, start_col:stop_col] == batch[:, None, start_col:stop_col]).all(dim=-1)
            batch_indices, triple_indices = triple_mask.nonzero(as_tuple=True)
            entity_indices = triples[triple_indices, sel_col]

            # shape: (batch_size, num_entities)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[batch_indices, entity_indices] = True
        else:
            mask = None

        return batch, scores, mask

    def test_process_object_scores_(self) -> None:
        """Test the evaluator's ``process_object_scores_()`` function."""
        batch, scores, mask = self._get_input()
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, 2]][:, None]
        self.evaluator.process_object_scores_(
            batch=batch,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

    def test_process_subject_scores_(self) -> None:
        """Test the evaluator's ``process_subject_scores_()`` function."""
        batch, scores, mask = self._get_input(inverse=True)
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, 0]][:, None]
        self.evaluator.process_subject_scores_(
            batch=batch,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

    def test_finalize(self) -> None:
        # Process one batch
        batch, scores, mask = self._get_input()
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, 2]][:, None]
        self.evaluator.process_object_scores_(
            batch=batch,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

        result = self.evaluator.finalize()
        assert isinstance(result, MetricResults)

        self._validate_result(
            result=result,
            data={'batch': batch, 'scores': scores, 'mask': mask}
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
        assert 1 <= result.mean_rank <= self.factory.num_entities
        assert 0 < result.mean_reciprocal_rank <= 1
        for k, v in result.hits_at_k.items():
            assert 0 <= v <= 1

        # TODO: Validate with data?


class _SklearnEvaluatorTests(_AbstractEvaluatorTests):
    """unittest for the SklearnEvaluator.."""

    evaluator_cls = SklearnEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, SklearnMetricResults)

        # check for correct name
        assert result.name == self.evaluator_kwargs['metric']

        # check value
        scores = data['scores'].detach().numpy()
        mask = data['mask'].detach().float().numpy()

        # filtering
        uniq = dict()
        batch = data['batch'].detach().numpy()
        for i, (s, p) in enumerate(batch[:, :2]):
            uniq[int(s), int(p)] = i
        indices = sorted(uniq.values())
        mask = mask[indices]
        scores = scores[indices]

        exp_score = self.__class__.metric(mask.flat, scores.flat)
        self.assertAlmostEqual(result.score, exp_score)


class ROCAUCEvaluatorTests(_SklearnEvaluatorTests, unittest.TestCase):
    """unittest for the SklearnEvaluator with roc_auc_score."""

    evaluator_kwargs = {'metric': 'roc_auc_score'}
    metric = roc_auc_score


class APSEvaluatorTests(_SklearnEvaluatorTests, unittest.TestCase):
    """unittest for the SklearnEvaluator with average_precision_score."""

    evaluator_kwargs = {'metric': 'average_precision_score'}
    metric = average_precision_score


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
        exp_avg_rank = torch.tensor([3.5, 2., 1.])
        exp_adj_rank = exp_avg_rank / torch.tensor([(5 + 1) / 2, (5 + 1) / 2, (4 + 1) / 2])
        avg_rank, adj_rank = compute_rank_from_scores(true_score=true_score, all_scores=all_scores)
        assert avg_rank.shape == (batch_size,)
        assert adj_rank.shape == (batch_size,)
        assert (avg_rank == exp_avg_rank).all(), (avg_rank, exp_avg_rank)
        assert (adj_rank == exp_adj_rank).all(), (adj_rank, exp_adj_rank)

    def test_create_sparse_positive_filter_(self):
        """Test method create_sparse_positive_filter_."""
        batch_size = 4
        factory = NationsTrainingTriplesFactory()
        all_triples = factory.mapped_triples
        batch = all_triples[:batch_size, :]

        # subject based filter
        sparse_positives, relation_filter = create_sparse_positive_filter_(
            batch=batch,
            all_pos_triples=all_triples,
            relation_filter=None,
            filter_col=0
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
        filter_batch[:, 1] = torch.randperm(num_positives)
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
        subject_filter_mask = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
            ], dtype=torch.bool,
        )
        object_filter_mask = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, True],
            ], dtype=torch.bool,
        )
        exp_subject_filter_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
            ], dtype=torch.bool,
        )
        exp_object_filter_mask = torch.tensor(
            [
                [False, False, True, False],
                [True, False, False, True],
            ], dtype=torch.bool,
        )
        assert batch.shape == (batch_size, 3)
        assert subject_filter_mask.shape == (batch_size, num_entities)
        assert object_filter_mask.shape == (batch_size, num_entities)

        # Test subject scores
        subject_scores = torch.randn(batch_size, num_entities)
        old_subject_scores = subject_scores.detach().clone()
        positive_filter_subjects, relation_filter = create_sparse_positive_filter_(
            batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=None,
            filter_col=0,
        )
        filtered_subject_scores = filter_scores_(
            scores=subject_scores,
            filter_batch=positive_filter_subjects,
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
        positive_filter_objects, _ = create_sparse_positive_filter_(
            batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=2,
        )
        filtered_object_scores = filter_scores_(
            scores=object_scores,
            filter_batch=positive_filter_objects,
        )
        # Assert in-place modification
        mask = torch.isfinite(object_scores)
        assert (object_scores[mask] == filtered_object_scores[mask]).all()
        assert not torch.isfinite(filtered_object_scores[~mask]).any()

        # Assert correct filtering
        assert (old_object_scores[~exp_object_filter_mask] == filtered_object_scores[~exp_object_filter_mask]).all()
        assert not torch.isfinite(filtered_object_scores[exp_object_filter_mask]).any()


class DummyEvaluator(Evaluator):
    """A dummy evaluator for testing the structure of the evaluation function."""

    def __init__(self, *, counter: int, filtered: bool) -> None:
        super().__init__(filtered=filtered)
        self.counter = counter

    def process_object_scores_(
        self,
        batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.BoolTensor] = None,
    ) -> None:  # noqa: D102
        self.counter += 1

    def process_subject_scores_(
        self,
        batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.BoolTensor] = None,
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


class DummyModel(BaseModule):
    """A dummy model returning fake scores."""

    def __init__(self, triples_factory: TriplesFactory):
        super().__init__(triples_factory=triples_factory)
        num_entities = self.num_entities
        self.scores = torch.arange(num_entities, dtype=torch.float)

    def _generate_fake_scores(self, batch: torch.LongTensor) -> torch.FloatTensor:
        """Generate fake scores s[b, i] = i of size (batch_size, num_entities)."""
        batch_size = batch.shape[0]
        batch_scores = self.scores.view(1, -1).repeat(batch_size, 1)
        assert batch_scores.shape == (batch_size, self.num_entities)
        return batch_scores

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=batch)

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=batch)

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._generate_fake_scores(batch=batch)

    def init_empty_weights_(self) -> BaseModule:  # noqa: D102
        raise NotImplementedError('Not needed for unittest')

    def clear_weights_(self) -> BaseModule:  # noqa: D102
        raise NotImplementedError('Not needed for unittest')


class TestEvaluationStructure(unittest.TestCase):
    """Tests for testing the correct structure of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation structure."""
        self.counter = 1337
        self.evaluator = DummyEvaluator(counter=self.counter, filtered=True)
        self.triples_factory = NationsTrainingTriplesFactory()
        self.model = DummyModel(triples_factory=self.triples_factory)

    def test_evaluation_structure(self):
        """Test if the evaluator has a balanced call of subject and object processors."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.triples_factory.mapped_triples,
            batch_size=1,
        )
        assert eval_results.mean_rank == self.counter, 'Should end at the same value as it started'
