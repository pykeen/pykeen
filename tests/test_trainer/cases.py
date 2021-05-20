# -*- coding: utf-8 -*-

"""Test cases for training."""

from collections import MutableMapping
from typing import Any, ClassVar, Type

import unittest_templates
from torch.optim import Adam, Optimizer

from pykeen.datasets import Nations
from pykeen.losses import CrossEntropyLoss
from pykeen.models import ConvE, Model, TransE
from pykeen.training import TrainingLoop
from pykeen.training.training_loop import TrainingApproachLossMismatchError
from pykeen.triples import TriplesFactory

__all__ = [
    'TrainingLoopTestCase',
    'SLCWATrainingLoopTestCase',
]


class TrainingLoopTestCase(unittest_templates.GenericTestCase[TrainingLoop]):
    """A generic test case for training loops."""

    model: Model
    factory: TriplesFactory
    optimizer_cls: ClassVar[Type[Optimizer]] = Adam
    optimizer: Optimizer
    random_seed = 0
    batch_size: int = 128
    sub_batch_size: int = 30

    def pre_setup_hook(self) -> None:
        self.triples_factory = Nations().training
        self.model = TransE(triples_factory=self.triples_factory, random_seed=self.random_seed)
        self.optimizer = self.optimizer_cls(self.model.get_grad_params())

    def _with_model(self, model: Model) -> TrainingLoop:
        return self.cls(
            model=model,
            triples_factory=self.triples_factory,
            automatic_memory_optimization=False,
            optimizer=self.optimizer_cls(model.get_grad_params()),
        )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = self.triples_factory
        kwargs["automatic_memory_optimization"] = False
        kwargs["optimizer"] = self.optimizer
        kwargs["model"] = self.model
        return kwargs

    def test_train(self):
        """Test training does not error."""
        self.instance.train(
            triples_factory=self.triples_factory,
            num_epochs=1,
        )

    def test_sub_batching(self):
        """Test if sub-batching works as expected."""
        self.instance.train(
            triples_factory=self.triples_factory,
            num_epochs=1,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
        )

    def test_sub_batching_support(self):
        """Test if sub-batching works as expected."""
        model = ConvE(triples_factory=self.triples_factory)
        training_loop = self._with_model(model)

        with self.assertRaises(NotImplementedError):
            training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=1,
                batch_size=self.batch_size,
                sub_batch_size=self.sub_batch_size,
            )


class SLCWATrainingLoopTestCase(TrainingLoopTestCase):
    """A generic test case for sLCWA training loops."""

    #: Should negative samples be filtered?
    filtered: ClassVar[bool]

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["negative_sampler"] = "basic"
        kwargs["negative_sampler_kwargs"] = {"filtered": self.filtered}
        return kwargs

    def test_blacklist_loss_on_slcwa(self):
        """Test an allowed sLCWA loss."""
        model = TransE(
            triples_factory=self.triples_factory,
            loss=CrossEntropyLoss(),
        )
        with self.assertRaises(TrainingApproachLossMismatchError):
            self.cls(
                model=model,
                triples_factory=self.triples_factory,
                automatic_memory_optimization=False,
            )
