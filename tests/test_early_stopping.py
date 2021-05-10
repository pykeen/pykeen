# -*- coding: utf-8 -*-

"""Tests of early stopping."""

import unittest
from typing import List

import numpy
import pytest
import torch
from torch.optim import Adam

from pykeen.datasets import Nations
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import Model, TransE
from pykeen.stoppers.early_stopping import EarlyStopper, is_improvement
from pykeen.trackers import MLFlowResultTracker
from pykeen.training import SLCWATrainingLoop
from tests.mocks import MockEvaluator, MockModel

try:
    import mlflow
except ImportError:
    mlflow = None


class TestRandom(unittest.TestCase):
    """Random tests for early stopper."""

    def test_is_improvement(self):
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
            with self.subTest(
                best_value=best_value,
                current_value=current_value,
                larger_is_better=larger_is_better,
                relative_delta=relative_delta,
                is_better=is_better,
            ):
                self.assertEqual(is_better, is_improvement(
                    best_value=best_value,
                    current_value=current_value,
                    larger_is_better=larger_is_better,
                    relative_delta=relative_delta,
                ))


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
        # Set automatic_memory_optimization to false for tests
        self.mock_evaluator = MockEvaluator(self.mock_losses, automatic_memory_optimization=False)
        nations = Nations()
        self.model = MockModel(triples_factory=nations.training)
        self.stopper = EarlyStopper(
            model=self.model,
            evaluator=self.mock_evaluator,
            training_triples_factory=nations.training,
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

    @unittest.skipUnless(mlflow is not None, reason='MLFlow not installed')
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

    @pytest.mark.slow
    def test_early_stopping(self):
        """Tests early stopping."""
        # Set automatic_memory_optimization to false during testing
        nations = Nations()
        model: Model = TransE(triples_factory=nations.training)
        evaluator = RankBasedEvaluator(automatic_memory_optimization=False)
        stopper = EarlyStopper(
            model=model,
            evaluator=evaluator,
            training_triples_factory=nations.training,
            evaluation_triples_factory=nations.validation,
            patience=self.patience,
            relative_delta=self.relative_delta,
            metric='mean_rank',
        )
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=nations.training,
            optimizer=Adam(params=model.get_grad_params()),
            automatic_memory_optimization=False,
        )
        losses = training_loop.train(
            triples_factory=nations.training,
            num_epochs=self.max_num_epochs,
            batch_size=self.batch_size,
            stopper=stopper,
            use_tqdm=False,
        )
        self.assertEqual(stopper.number_results, (len(losses) + self.patience * stopper.frequency) // stopper.frequency)
        self.assertEqual(
            self.stop_epoch,
            (len(losses) + 2 * stopper.frequency),
            msg='Did not stop early like it should have',
        )
