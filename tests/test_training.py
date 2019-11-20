# -*- coding: utf-8 -*-

"""Test that training loops work correctly."""

import unittest

import torch
from torch import optim

from poem.datasets import NationsTrainingTriplesFactory
from poem.models import BaseModule, TransE
from poem.training import OWATrainingLoop
from poem.typing import MappedTriples


class DummyTrainingLoop(OWATrainingLoop):
    """A wrapper around OWATrainingLoop."""

    def __init__(self, model: BaseModule, sub_batch_size: int):
        super().__init__(model=model, optimizer=optim.Adam(lr=1.0, params=model.parameters()))
        self.sub_batch_size = sub_batch_size

    def _process_batch(
        self,
        batch: MappedTriples,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
    ) -> torch.FloatTensor:  # noqa: D102
        assert (stop - start) <= self.sub_batch_size
        return super()._process_batch(batch=batch, start=start, stop=stop, label_smoothing=label_smoothing)


class TrainingLoopTests(unittest.TestCase):
    """Tests for the general training loop."""

    batch_size: int = 128
    sub_batch_size: int = 30

    def setUp(self) -> None:
        """Instantiate triples factory and model."""
        self.triples_factory = NationsTrainingTriplesFactory()
        self.model = TransE(triples_factory=self.triples_factory)

    def test_subbatching(self):
        """Test if sub-batching works as expected."""
        training_loop = DummyTrainingLoop(model=self.model, sub_batch_size=self.sub_batch_size)
        training_loop.train(num_epochs=1, batch_size=self.batch_size, sub_batch_size=self.sub_batch_size)
