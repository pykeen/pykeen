# -*- coding: utf-8 -*-

"""Test that training loops work correctly."""

import unittest
from typing import List, Optional

import torch
from torch import optim

from pykeen.datasets import Nations
from pykeen.models import Model
from pykeen.stoppers.early_stopping import EarlyStopper
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.typing import MappedTriples
from tests.mocks import MockEvaluator, MockModel


class DummyTrainingLoop(SLCWATrainingLoop):
    """A wrapper around SLCWATrainingLoop."""

    def __init__(
        self,
        model: Model,
        triples_factory: TriplesFactory,
        sub_batch_size: int,
        automatic_memory_optimization: bool = False,
    ):
        super().__init__(
            model=model,
            triples_factory=triples_factory,
            optimizer=optim.Adam(lr=1.0, params=model.parameters()),
            automatic_memory_optimization=automatic_memory_optimization,
        )
        self.sub_batch_size = sub_batch_size

    def _process_batch(
        self,
        batch: MappedTriples,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        assert (stop - start) <= self.sub_batch_size

        # check for empty batches
        if torch.is_tensor(batch):
            assert batch[start:stop].shape[0] > 0

        return super()._process_batch(
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
        )


class TestTrainingEarlyStopping(unittest.TestCase):
    """Tests for early stopping during training."""

    batch_size: int = 128
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
        self.triples_factory = Nations()
        self.model = MockModel(triples_factory=self.triples_factory.training)
        self.stopper = EarlyStopper(
            model=self.model,
            evaluator=self.mock_evaluator,
            training_triples_factory=self.triples_factory.training,
            evaluation_triples_factory=self.triples_factory.validation,
            patience=self.patience,
            relative_delta=self.delta,
            larger_is_better=False,
            frequency=1,
        )

    def test_early_stopper_best_epoch_model_retrieval(self):
        """Test if the best epoch model is returned when using the early stopper."""
        training_loop = DummyTrainingLoop(
            model=self.model,
            triples_factory=self.triples_factory.training,
            sub_batch_size=self.batch_size,
        )

        _ = training_loop.train(
            triples_factory=self.triples_factory.training,
            num_epochs=10,
            batch_size=self.batch_size,
            stopper=self.stopper,
        )
        self.assertEqual(training_loop._epoch, len(self.stopper.results) - self.patience)
