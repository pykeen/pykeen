# -*- coding: utf-8 -*-

"""Test that training loops work correctly."""

import tempfile
import unittest
from typing import List, Optional

import torch
from torch import optim

from pykeen.datasets import Nations
from pykeen.losses import CrossEntropyLoss
from pykeen.models import ConvE, Model, TransE
from pykeen.optimizers import optimizer_resolver
from pykeen.stoppers.early_stopping import EarlyStopper
from pykeen.training import SLCWATrainingLoop, training_loop_resolver
from pykeen.training.training_loop import NonFiniteLossError, TrainingApproachLossMismatchError
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


class NaNTrainingLoop(SLCWATrainingLoop):
    """A wrapper around SLCWATrainingLoop returning NaN losses."""

    def __init__(
        self,
        model: Model,
        triples_factory: TriplesFactory,
        patience: int,
        automatic_memory_optimization: bool = False,
    ):
        super().__init__(
            model=model,
            triples_factory=triples_factory,
            optimizer=optim.Adam(lr=1.0, params=model.parameters()),
            automatic_memory_optimization=automatic_memory_optimization,
        )
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
        self.random_seed = 123
        self.checkpoint_file = "PyKEEN_training_loop_test_checkpoint.pt"
        self.num_epochs = 10
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Tear down the test case."""
        self.temporary_directory.cleanup()

    def test_sub_batching(self):
        """Test if sub-batching works as expected."""
        model = TransE(triples_factory=self.triples_factory)
        training_loop = DummyTrainingLoop(
            model=model,
            triples_factory=self.triples_factory,
            sub_batch_size=self.sub_batch_size,
            automatic_memory_optimization=False,
        )
        training_loop.train(
            triples_factory=self.triples_factory,
            num_epochs=1,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
        )

    def test_sub_batching_support(self):
        """Test if sub-batching works as expected."""
        model = ConvE(triples_factory=self.triples_factory)
        training_loop = DummyTrainingLoop(
            model=model,
            triples_factory=self.triples_factory,
            sub_batch_size=self.sub_batch_size,
            automatic_memory_optimization=False,
        )

        def _try_train():
            """Call train method."""
            training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=1,
                batch_size=self.batch_size,
                sub_batch_size=self.sub_batch_size,
            )

        self.assertRaises(NotImplementedError, _try_train)

    def test_error_on_nan(self):
        """Test if the correct error is raised for non-finite loss values."""
        model = TransE(triples_factory=self.triples_factory)
        training_loop = NaNTrainingLoop(model=model, triples_factory=self.triples_factory, patience=2)

        with self.assertRaises(NonFiniteLossError):
            training_loop.train(triples_factory=self.triples_factory, num_epochs=3, batch_size=self.batch_size)

    def test_blacklist_loss_on_slcwa(self):
        """Test an allowed sLCWA loss."""
        model = TransE(
            triples_factory=self.triples_factory,
            loss=CrossEntropyLoss(),
        )
        with self.assertRaises(TrainingApproachLossMismatchError):
            NaNTrainingLoop(
                model=model,
                triples_factory=self.triples_factory,
                patience=2,
                automatic_memory_optimization=False,
            )

    def test_lcwa_checkpoints(self):
        """Test whether interrupting the LCWA training loop can be resumed using checkpoints."""
        self._test_checkpoints(training_loop_type='LCWA')

    def test_slcwa_checkpoints(self):
        """Test whether interrupting the sLCWA training loop can be resumed using checkpoints."""
        self._test_checkpoints(training_loop_type='sLCWA')

    def _test_checkpoints(self, training_loop_type: str):
        """Test whether interrupting the given training loop type can be resumed using checkpoints."""
        training_loop_class = training_loop_resolver.lookup(training_loop_type)

        # Train a model in one shot
        model = TransE(
            triples_factory=self.triples_factory,
            random_seed=self.random_seed,
        )
        optimizer_cls = optimizer_resolver.lookup(None)
        optimizer = optimizer_cls(params=model.get_grad_params())
        training_loop = training_loop_class(
            model=model,
            triples_factory=self.triples_factory,
            optimizer=optimizer,
            automatic_memory_optimization=False,
        )
        losses = training_loop.train(
            triples_factory=self.triples_factory,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_tqdm=False,
            use_tqdm_batch=False,
        )

        # Train a model for the first half
        model = TransE(
            triples_factory=self.triples_factory,
            random_seed=self.random_seed,
        )
        optimizer_cls = optimizer_resolver.lookup(None)
        optimizer = optimizer_cls(params=model.get_grad_params())
        training_loop = training_loop_class(
            model=model,
            triples_factory=self.triples_factory,
            optimizer=optimizer,
            automatic_memory_optimization=False,
        )
        training_loop.train(
            triples_factory=self.triples_factory,
            num_epochs=int(self.num_epochs // 2),
            batch_size=self.batch_size,
            checkpoint_name=self.checkpoint_file,
            checkpoint_directory=self.temporary_directory.name,
            checkpoint_frequency=0,
        )

        # Continue training of the first part
        model = TransE(
            triples_factory=self.triples_factory,
            random_seed=123,
        )
        optimizer_cls = optimizer_resolver.lookup(None)
        optimizer = optimizer_cls(params=model.get_grad_params())
        training_loop = training_loop_class(
            model=model,
            triples_factory=self.triples_factory,
            optimizer=optimizer,
            automatic_memory_optimization=False,
        )
        losses_2 = training_loop.train(
            triples_factory=self.triples_factory,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            checkpoint_name=self.checkpoint_file,
            checkpoint_directory=self.temporary_directory.name,
            checkpoint_frequency=0,
        )

        self.assertEqual(losses, losses_2)


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
