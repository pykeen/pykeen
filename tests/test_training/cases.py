# -*- coding: utf-8 -*-

"""Test cases for training."""

import tempfile
from typing import Any, ClassVar, MutableMapping, Optional, Type

import torch
import unittest_templates
from torch.optim import Adam, Optimizer

from pykeen.datasets import Nations
from pykeen.losses import Loss
from pykeen.models import ConvE, Model, TransE
from pykeen.sampling.filtering import Filterer
from pykeen.trackers.base import PythonResultTracker
from pykeen.training import TrainingLoop
from pykeen.training.training_loop import NonFiniteLossError, NoTrainingBatchError
from pykeen.triples import TriplesFactory

__all__ = [
    "TrainingLoopTestCase",
    "SLCWATrainingLoopTestCase",
]


class TrainingLoopTestCase(unittest_templates.GenericTestCase[TrainingLoop]):
    """A generic test case for training loops."""

    model: Model
    factory: TriplesFactory
    loss_cls: ClassVar[Type[Loss]]
    loss: Loss
    optimizer_cls: ClassVar[Type[Optimizer]] = Adam
    optimizer: Optimizer
    random_seed = 0
    batch_size: int = 128
    sub_batch_size: int = 30
    num_epochs: int = 10

    def pre_setup_hook(self) -> None:
        """Prepare case-level variables before the setup() function."""
        self.triples_factory = Nations().training
        self.loss = self.loss_cls()
        self.model = TransE(triples_factory=self.triples_factory, loss=self.loss, random_seed=self.random_seed)
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

    def test_error_on_nan(self):
        """Test if the correct error is raised for non-finite loss values."""
        model = TransE(triples_factory=self.triples_factory)
        patience = 2

        class NaNTrainingLoop(self.cls):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.patience = patience

            def _process_batch(self, *args, **kwargs):
                loss = super()._process_batch(*args, **kwargs)
                self.patience -= 1
                if self.patience < 0:
                    return torch.as_tensor([float("nan")], device=loss.device, dtype=torch.float32)
                return loss

        training_loop = NaNTrainingLoop(
            model=model,
            triples_factory=self.triples_factory,
            optimizer=self.optimizer_cls(model.get_grad_params()),
        )
        with self.assertRaises(NonFiniteLossError):
            training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=patience + 1,
                batch_size=self.batch_size,
            )

    def test_checkpoints(self):
        """Test whether interrupting the given training loop type can be resumed using checkpoints."""
        # Train a model in one shot
        model = TransE(
            triples_factory=self.triples_factory,
            random_seed=self.random_seed,
        )
        training_loop = self._with_model(model)
        losses = training_loop.train(
            triples_factory=self.triples_factory,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_tqdm=False,
            use_tqdm_batch=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            name = "checkpoint.pt"

            # Train a model for the first half
            model = TransE(
                triples_factory=self.triples_factory,
                random_seed=self.random_seed,
            )
            training_loop = self._with_model(model)
            training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=int(self.num_epochs // 2),
                batch_size=self.batch_size,
                checkpoint_name=name,
                checkpoint_directory=directory,
                checkpoint_frequency=0,
            )

            # Continue training of the first part
            model = TransE(
                triples_factory=self.triples_factory,
                random_seed=123,
            )
            training_loop = self._with_model(model)
            losses_2 = training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                checkpoint_name=name,
                checkpoint_directory=directory,
                checkpoint_frequency=0,
            )

        self.assertEqual(losses, losses_2)

    def test_result_tracker(self):
        """Test whether losses are tracked by the result tracker."""
        self.instance.result_tracker = PythonResultTracker()
        self.instance.train(
            triples_factory=self.triples_factory,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )
        # check non-empty metrics
        assert self.instance.result_tracker.metrics

    def test_error_on_no_batch(self):
        """Verify that an error is raised if no training batch is available."""
        with self.assertRaises(NoTrainingBatchError):
            self.instance.train(
                triples_factory=self.triples_factory,
                num_epochs=self.num_epochs,
                drop_last=True,
                batch_size=100_000,
            )


class SLCWATrainingLoopTestCase(TrainingLoopTestCase):
    """A generic test case for sLCWA training loops."""

    #: Should negative samples be filtered?
    filterer_cls: ClassVar[Optional[Type[Filterer]]] = None

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["negative_sampler"] = "basic"
        kwargs["negative_sampler_kwargs"] = {"filterer": self.filterer_cls}
        return kwargs
