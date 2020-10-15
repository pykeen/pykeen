# -*- coding: utf-8 -*-

"""Test that training loops work correctly."""

import unittest
from typing import Optional

import torch
from torch import optim

from pykeen.datasets import Nations
from pykeen.losses import CrossEntropyLoss
from pykeen.models import ConvE, TransE
from pykeen.models.base import Model
from pykeen.training import SLCWATrainingLoop
from pykeen.training.training_loop import NonFiniteLossError, TrainingApproachLossMismatchError
from pykeen.typing import MappedTriples


class DummyTrainingLoop(SLCWATrainingLoop):
    """A wrapper around SLCWATrainingLoop."""

    def __init__(self, model: Model, sub_batch_size: int):
        super().__init__(model=model, optimizer=optim.Adam(lr=1.0, params=model.parameters()))
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


class NaNTrainingLoop(SLCWATrainingLoop):
    """A wrapper around SLCWATrainingLoop returning NaN losses."""

    def __init__(self, model: Model, patience: int):
        super().__init__(model=model, optimizer=optim.Adam(lr=1.0, params=model.parameters()))
        self.patience = patience

    def _process_batch(
        self,
        batch: MappedTriples,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self.patience -= 1
        if self.patience < 0:
            return torch.as_tensor([float('nan')], device=batch.device, dtype=torch.float32)
        else:
            factor = 1.0
        loss = super()._process_batch(
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
        )
        return factor * loss


class TrainingLoopTests(unittest.TestCase):
    """Tests for the general training loop."""

    batch_size: int = 128
    sub_batch_size: int = 30

    def setUp(self) -> None:
        """Instantiate triples factory and model."""
        self.triples_factory = Nations().training

    def test_sub_batching(self):
        """Test if sub-batching works as expected."""
        model = TransE(triples_factory=self.triples_factory, automatic_memory_optimization=False)
        training_loop = DummyTrainingLoop(model=model, sub_batch_size=self.sub_batch_size)
        training_loop.train(num_epochs=1, batch_size=self.batch_size, sub_batch_size=self.sub_batch_size)

    def test_sub_batching_support(self):
        """Test if sub-batching works as expected."""
        model = ConvE(triples_factory=self.triples_factory, automatic_memory_optimization=False)
        training_loop = DummyTrainingLoop(model=model, sub_batch_size=self.sub_batch_size)

        def _try_train():
            """Call train method."""
            training_loop.train(num_epochs=1, batch_size=self.batch_size, sub_batch_size=self.sub_batch_size)

        self.assertRaises(NotImplementedError, _try_train)

    def test_error_on_nan(self):
        """Test if the correct error is raised for non-finite loss values."""
        model = TransE(triples_factory=self.triples_factory)
        training_loop = NaNTrainingLoop(model=model, patience=2)

        with self.assertRaises(NonFiniteLossError):
            training_loop.train(num_epochs=3, batch_size=self.batch_size)

    def test_blacklist_loss_on_slcwa(self):
        """Test an allowed sLCWA loss."""
        model = TransE(
            triples_factory=self.triples_factory,
            loss=CrossEntropyLoss(),
            automatic_memory_optimization=False,
        )
        with self.assertRaises(TrainingApproachLossMismatchError):
            NaNTrainingLoop(model=model, patience=2)
