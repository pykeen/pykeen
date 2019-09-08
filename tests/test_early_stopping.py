# -*- coding: utf-8 -*-

"""Tests of early stopping."""

import unittest
from typing import Iterable, List

import numpy
import torch
from torch.optim import Adam

from poem.datasets.nations import NationsTrainingTriplesFactory, NationsValidationTriplesFactory
from poem.evaluation import Evaluator, MetricResults, RankBasedEvaluator
from poem.models import TransE
from poem.models.base import BaseModule
from poem.training import EarlyStopper, OWATrainingLoop
from poem.training.early_stopping import larger_than_any_buffer_element, smaller_than_any_buffer_element
from poem.typing import MappedTriples


class TestImprovementChecking(unittest.TestCase):
    """Tests for checking improvement."""

    def test_smaller_than_any_buffer_element(self):
        """Test ``smaller_than_any_buffer_element``."""
        buffer = numpy.asarray([1.0, 0.9, 0.8])
        assert not smaller_than_any_buffer_element(buffer=buffer, result=1.0)
        assert smaller_than_any_buffer_element(buffer=buffer, result=0.9)
        assert not smaller_than_any_buffer_element(buffer=buffer, result=0.9, delta=0.1)

    def test_larger_than_any_buffer_element(self):
        """Test ``smaller_than_any_buffer_element``."""
        buffer = numpy.asarray([1.0, 0.9, 0.8])
        assert larger_than_any_buffer_element(buffer=buffer, result=1.0)
        assert larger_than_any_buffer_element(buffer=buffer, result=1.0, delta=0.1)
        assert not larger_than_any_buffer_element(buffer=buffer, result=0.9, delta=0.1)


class MockEvaluator(Evaluator):
    """A mock evaluator for testing early stopping."""

    def __init__(self, losses: Iterable[float]) -> None:
        super().__init__(None)
        self.losses = tuple(losses)
        self.losses_iter = iter(self.losses)

    def __repr__(self):  # noqa: D105
        return f'MockEvaluator(losses={self.losses})'

    def evaluate(self, mapped_triples: MappedTriples, **kwargs) -> MetricResults:
        """Return a metric package with the next loss."""
        return MetricResults(
            mean_rank=None,
            mean_reciprocal_rank=None,
            adjusted_mean_rank=None,
            adjusted_mean_reciprocal_rank=None,
            hits_at_k={
                10: next(self.losses_iter),
            },
        )


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

    def setUp(self):
        """Prepare for testing the early stopper."""
        self.mock_evaluator = MockEvaluator(self.mock_losses)
        self.early_stopper = EarlyStopper(
            evaluator=self.mock_evaluator,
            evaluation_triples_factory=NationsValidationTriplesFactory(),
            patience=self.patience,
            delta=self.delta,
            larger_is_better=False,
        )

    def test_initialization(self):
        """Test warm-up phase."""
        for it in range(self.patience):
            should_stop = self.early_stopper.should_stop()
            assert self.early_stopper.number_evaluations == it + 1
            assert not should_stop

    def test_result_processing(self):
        """Test that the mock evaluation of the early stopper always gives the right loss."""
        for stop, loss in enumerate(self.mock_losses, start=1):
            # Step early stopper
            should_stop = self.early_stopper.should_stop()

            if not should_stop:
                # check storing of results
                assert self.early_stopper.results == self.mock_losses[:stop]

                # check ring buffer
                if stop >= self.patience:
                    assert set(self.early_stopper.buffer) == set(self.mock_losses[stop - self.patience:stop])

    def test_should_stop(self):
        """Test that the stopper knows when to stop."""
        for _ in range(self.stop_constant):
            self.assertFalse(self.early_stopper.should_stop())
        self.assertTrue(self.early_stopper.should_stop())


class TestDeltaEarlyStopping(TestEarlyStopping):
    """Test early stopping with a tiny delta."""

    mock_losses: List[float] = [10.0, 9.0, 8.0, 7.99, 7.98, 7.97]
    stop_constant: int = 4
    delta: float = 0.1


class TestEarlyStoppingRealWorld(unittest.TestCase):
    """Test early stopping on a real-world use case of training TransE with Adam."""

    #: The window size used by the early stopper
    patience: int = 2
    #: The (zeroed) index  - 1 at which stopping will occur
    stop_constant: int = 4
    #: The minimum improvement
    delta: float = 0.1
    #: The random seed to use for reproducibility
    seed: int = 42
    #: The maximum number of epochs to train. Should be large enough to allow for early stopping.
    max_num_epochs: int = 1000
    #: The epoch at which the stop should happen. Depends on the choice of random seed.
    stop_epoch: int = 21
    #: The batch size to use.
    batch_size: int = 128

    def setUp(self) -> None:
        """Set up the real world early stopping test."""
        # Fix seed for reproducibility
        torch.manual_seed(seed=self.seed)
        numpy.random.seed(seed=self.seed)

    def test_early_stopping(self):
        """Tests early stopping."""
        model: BaseModule = TransE(triples_factory=NationsTrainingTriplesFactory())
        evaluator = RankBasedEvaluator(model=model)
        early_stopper = EarlyStopper(
            evaluator=evaluator,
            evaluation_triples_factory=NationsValidationTriplesFactory(),
            patience=self.patience,
            delta=self.delta,
            metric='mean_rank',
        )
        training_loop = OWATrainingLoop(
            model=model,
            optimizer=Adam(params=model.get_grad_params()),
        )
        losses = training_loop.train(
            num_epochs=self.max_num_epochs,
            batch_size=self.batch_size,
            early_stopper=early_stopper,
        )
        assert len(early_stopper.results) == len(losses) // early_stopper.frequency
        self.assertEqual(self.stop_epoch, len(losses), msg='Did not stop early like it should have')
