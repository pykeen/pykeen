# -*- coding: utf-8 -*-

"""Tests of early stopping."""

import unittest
from typing import Iterable, List

import numpy as np
from torch.optim import Adagrad

from poem.datasets.nations import NationsTrainingTriplesFactory
from poem.evaluation import Evaluator, MetricResults
from poem.models import TransE
from poem.training import EarlyStopper, OWATrainingLoop


class MockEvaluator(Evaluator):
    """A mock evaluator for testing early stopping."""

    def __init__(self, losses: Iterable[float]) -> None:
        super().__init__(None)
        self.losses = tuple(losses)
        self.losses_iter = iter(self.losses)

    def __repr__(self):  # noqa: D105
        return f'MockEvaluator(losses={self.losses})'

    def evaluate(self, triples) -> MetricResults:
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
    window: int = 2
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
            window=self.window,
            triples=np.ndarray([]),
            delta=self.delta,
        )

    def test_initialization(self):
        """Test that the early stopper is initialized after being evaluated ``window + 1`` times."""
        for _ in range(self.window + 1):
            self.assertFalse(self.early_stopper.initialized)
            self.early_stopper.evaluate()
        self.assertTrue(self.early_stopper.initialized)

    def test_current_loss(self):
        """Test that the mock evaluation of the early stopper always gives the right loss."""
        for i, loss in enumerate(self.mock_losses):
            self.assertEqual(i, len(self.early_stopper.results))
            self.assertEqual(loss, self.early_stopper.evaluate())

    def test_should_stop(self):
        """Test that the stopper knows when to stop."""
        for _ in range(self.stop_constant):
            self.assertFalse(self.early_stopper.should_stop())
        self.assertTrue(self.early_stopper.should_stop())

    @unittest.skip('Blocked by issues with models')
    def test_early_stopping(self):
        """Tests early stopping."""
        triple_factory = NationsTrainingTriplesFactory()
        model = TransE(triples_factory=triple_factory)
        optimizer = Adagrad(params=model.get_grad_params())
        training_loop = OWATrainingLoop(
            model=model,
            optimizer=optimizer,
        )

        losses = training_loop.train(
            num_epochs=10,
            batch_size=2,
            early_stopper=self.early_stopper,
        )
        self.assertEqual(5, len(losses), msg='Did not stop early like it should have')


class TestDeltaEarlyStopping(TestEarlyStopping):
    """Test early stopping with a tiny delta."""

    mock_losses: List[float] = [10.0, 9.0, 8.0, 7.99, 7.98, 7.97]
    stop_constant: int = 4
    delta: float = 0.1
