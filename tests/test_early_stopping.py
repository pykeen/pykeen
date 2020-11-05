# -*- coding: utf-8 -*-

"""Tests of early stopping."""

import unittest
from typing import Iterable, List, Optional

import numpy
import torch
from torch.optim import Adam

from pykeen.datasets import Nations
from pykeen.evaluation import Evaluator, MetricResults, RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.rank_based_evaluator import RANK_TYPES, SIDES
from pykeen.models import TransE
from pykeen.models.base import EntityRelationEmbeddingModel, Model
from pykeen.stoppers.early_stopping import EarlyStopper, is_improvement
from pykeen.trackers import MLFlowResultTracker
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.typing import MappedTriples


def test_is_improvement():
    """Test is_improvement()."""
    for best_value, current_value, larger_is_better, relative_delta, is_better in [
        # equal value; larger is better
        (1.0, 1.0, True, 0.0, False),
        # equal value; smaller is better
        (1.0, 1.0, False, 0.0, False),
        # larger is better; improvement
        (1.0, 1.1, True, 0.0, True),
        # larger is better; improvement; but not significant
        (1.0, 1.1, True, 0.1, False),
    ]:
        assert is_better == is_improvement(
            best_value=best_value,
            current_value=current_value,
            larger_is_better=larger_is_better,
            relative_delta=relative_delta,
        )


class MockEvaluator(Evaluator):
    """A mock evaluator for testing early stopping."""

    def __init__(self, losses: Iterable[float]) -> None:
        super().__init__()
        self.losses = tuple(losses)
        self.losses_iter = iter(self.losses)

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        pass

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        pass

    def finalize(self) -> MetricResults:  # noqa: D102
        hits = next(self.losses_iter)
        return RankBasedMetricResults(
            mean_rank={
                side: {
                    rank_type: 10
                    for rank_type in RANK_TYPES
                }
                for side in SIDES
            },
            mean_reciprocal_rank={
                side: {
                    rank_type: 1.0
                    for rank_type in RANK_TYPES
                }
                for side in SIDES
            },
            adjusted_mean_rank={
                side: 1.0
                for side in SIDES
            },
            hits_at_k={
                side: {
                    rank_type: {
                        10: hits,
                    } for rank_type in RANK_TYPES
                }
                for side in SIDES
            },
        )

    def __repr__(self):  # noqa: D105
        return f'{self.__class__.__name__}(losses={self.losses})'


class MockModel(EntityRelationEmbeddingModel):
    """A mock model returning fake scores."""

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


class LogCallWrapper:
    """An object which wraps functions and checks whether they have been called."""

    def __init__(self):
        self.called = set()

    def wrap(self, func):
        """Wrap the function."""
        id_func = id(func)

        def wrapped(*args, **kwargs):
            self.called.add(id_func)
            return func(*args, **kwargs)

        return wrapped

    def was_called(self, func) -> bool:
        """Report whether the previously wrapped function has been called."""
        return id(func) in self.called


class TestEarlyStopping(unittest.TestCase):
    """Tests for early stopping."""

    #: The window size used by the early stopper
    patience: int = 2
    #: The mock losses the mock evaluator will return
    mock_losses: List[float] = [10.0, 9.0, 8.0, 8.0, 8.0, 8.0]
    #: The (zeroed) index  - 1 at which stopping will occur
    stop_constant: int = 4
    #: The minimum improvement
    delta: float = 0.0
    #: The best results
    best_results: List[float] = [10.0, 9.0, 8.0, 8.0, 8.0]

    def setUp(self):
        """Prepare for testing the early stopper."""
        self.mock_evaluator = MockEvaluator(self.mock_losses)
        # Set automatic_memory_optimization to false for tests
        nations = Nations()
        self.model = MockModel(triples_factory=nations.training, automatic_memory_optimization=False)
        self.stopper = EarlyStopper(
            model=self.model,
            evaluator=self.mock_evaluator,
            evaluation_triples_factory=nations.validation,
            patience=self.patience,
            relative_delta=self.delta,
            larger_is_better=False,
        )

    def test_initialization(self):
        """Test warm-up phase."""
        for epoch in range(self.patience):
            should_stop = self.stopper.should_stop(epoch=epoch)
            assert not should_stop

    def test_result_processing(self):
        """Test that the mock evaluation of the early stopper always gives the right loss."""
        for epoch in range(len(self.mock_losses)):
            # Step early stopper
            should_stop = self.stopper.should_stop(epoch=epoch)

            if not should_stop:
                # check storing of results
                assert self.stopper.results == self.mock_losses[:epoch + 1]

                # check ring buffer
                if epoch >= self.patience:
                    assert self.stopper.best_metric == self.best_results[epoch]

    def test_should_stop(self):
        """Test that the stopper knows when to stop."""
        for epoch in range(self.stop_constant):
            self.assertFalse(self.stopper.should_stop(epoch=epoch))
        self.assertTrue(self.stopper.should_stop(epoch=epoch))

    def test_result_logging_with_mlflow(self):
        """Test whether the MLFLow result logger works."""
        self.stopper.result_tracker = MLFlowResultTracker()
        wrapper = LogCallWrapper()
        real_log_metrics = self.stopper.result_tracker.mlflow.log_metrics
        self.stopper.result_tracker.mlflow.log_metrics = wrapper.wrap(real_log_metrics)
        self.stopper.should_stop(epoch=0)
        assert wrapper.was_called(real_log_metrics)


class TestDeltaEarlyStopping(TestEarlyStopping):
    """Test early stopping with a tiny delta."""

    mock_losses: List[float] = [10.0, 9.0, 8.0, 7.99, 7.98, 7.97]
    stop_constant: int = 4
    delta: float = 0.1
    best_results: List[float] = [10.0, 9.0, 8.0, 8.0, 8.0]


class TestEarlyStoppingRealWorld(unittest.TestCase):
    """Test early stopping on a real-world use case of training TransE with Adam."""

    #: The window size used by the early stopper
    patience: int = 2
    #: The (zeroed) index  - 1 at which stopping will occur
    stop_constant: int = 4
    #: The minimum improvement
    relative_delta: float = 0.1
    #: The random seed to use for reproducibility
    seed: int = 42
    #: The maximum number of epochs to train. Should be large enough to allow for early stopping.
    max_num_epochs: int = 1000
    #: The epoch at which the stop should happen. Depends on the choice of random seed.
    stop_epoch: int = 30
    #: The batch size to use.
    batch_size: int = 128

    def setUp(self) -> None:
        """Set up the real world early stopping test."""
        # Fix seed for reproducibility
        torch.manual_seed(seed=self.seed)
        numpy.random.seed(seed=self.seed)

    def test_early_stopping(self):
        """Tests early stopping."""
        # Set automatic_memory_optimization to false during testing
        nations = Nations()
        model: Model = TransE(triples_factory=nations.training, automatic_memory_optimization=False)
        evaluator = RankBasedEvaluator()
        stopper = EarlyStopper(
            model=model,
            evaluator=evaluator,
            evaluation_triples_factory=nations.validation,
            patience=self.patience,
            relative_delta=self.relative_delta,
            metric='mean_rank',
        )
        training_loop = SLCWATrainingLoop(
            model=model,
            optimizer=Adam(params=model.get_grad_params()),
        )
        losses = training_loop.train(
            num_epochs=self.max_num_epochs,
            batch_size=self.batch_size,
            stopper=stopper,
        )
        self.assertEqual(stopper.number_results, len(losses) // stopper.frequency)
        self.assertEqual(self.stop_epoch, len(losses), msg='Did not stop early like it should have')
