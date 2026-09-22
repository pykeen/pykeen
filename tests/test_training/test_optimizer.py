"""Tests for the optimizer and LR scheduler handling of training loops."""

import gc
import weakref

import pytest
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

from pykeen.datasets import Nations
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.training import SLCWATrainingLoop, TrainingLoop
from pykeen.triples import CoreTriplesFactory


@pytest.fixture
def triples_factory() -> CoreTriplesFactory:
    """Return the training triples of a small dataset."""
    return Nations().training


def _train(training_loop: TrainingLoop, triples_factory: CoreTriplesFactory, **kwargs) -> None:
    training_loop.train(triples_factory=triples_factory, num_epochs=1, batch_size=256, use_tqdm=False, **kwargs)


def test_pipeline_adamw() -> None:
    """Test that the pipeline works with an optimizer whose ``defaults`` are not all ``__init__`` parameters."""
    pipeline(
        dataset="nations",
        model="transe",
        model_kwargs={"embedding_dim": 2},
        optimizer=AdamW,
        optimizer_kwargs={"weight_decay": 1.0e-04},
        training_kwargs={"num_epochs": 1},
    )


def test_fresh_run(triples_factory: CoreTriplesFactory) -> None:
    """Test that each fresh run re-creates the optimizer and LR scheduler from their hints."""
    model = TransE(triples_factory=triples_factory, embedding_dim=2)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer=AdamW,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0.2},
        lr_scheduler=StepLR,
        lr_scheduler_kwargs={"step_size": 3, "gamma": 0.5},
        automatic_memory_optimization=False,
    )
    optimizer = training_loop.optimizer
    lr_scheduler = training_loop.lr_scheduler

    # the first run uses the optimizer created on initialization
    _train(training_loop, triples_factory)
    assert training_loop.optimizer is optimizer
    assert training_loop.lr_scheduler is lr_scheduler

    _train(training_loop, triples_factory)
    new_optimizer = training_loop.optimizer
    assert isinstance(new_optimizer, AdamW)
    assert new_optimizer is not optimizer
    assert new_optimizer.defaults["lr"] == 0.1
    assert new_optimizer.defaults["weight_decay"] == 0.2
    new_lr_scheduler = training_loop.lr_scheduler
    assert isinstance(new_lr_scheduler, StepLR)
    assert new_lr_scheduler is not lr_scheduler
    assert new_lr_scheduler.optimizer is new_optimizer
    assert new_lr_scheduler.step_size == 3
    assert new_lr_scheduler.gamma == 0.5


def test_fresh_run_pre_instantiated(triples_factory: CoreTriplesFactory) -> None:
    """Test that a pre-instantiated optimizer is used for the first run, but not re-created for further fresh runs."""
    model = TransE(triples_factory=triples_factory, embedding_dim=2)
    optimizer = AdamW(params=model.get_grad_params(), lr=0.1, weight_decay=0.2)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=3, gamma=0.5)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        automatic_memory_optimization=False,
    )

    _train(training_loop, triples_factory)
    assert training_loop.optimizer is optimizer
    assert training_loop.lr_scheduler is lr_scheduler

    # continuing is fine
    _train(training_loop, triples_factory, continue_training=True)
    assert training_loop.optimizer is optimizer

    with pytest.raises(ValueError, match="pre-instantiated"):
        _train(training_loop, triples_factory)


def test_continue_training(triples_factory: CoreTriplesFactory) -> None:
    """Test that continuing training keeps the optimizer and its state."""
    model = TransE(triples_factory=triples_factory, embedding_dim=2)
    training_loop = SLCWATrainingLoop(
        model=model, triples_factory=triples_factory, optimizer=Adam, automatic_memory_optimization=False
    )
    _train(training_loop, triples_factory)
    optimizer = training_loop.optimizer
    assert optimizer is not None
    assert optimizer.state

    _train(training_loop, triples_factory, continue_training=True)
    assert training_loop.optimizer is optimizer


def test_clear_optimizer(triples_factory: CoreTriplesFactory) -> None:
    """Test that clearing the optimizer releases a pre-instantiated optimizer."""
    model = TransE(triples_factory=triples_factory, embedding_dim=2)
    optimizer = Adam(params=model.get_grad_params())
    reference = weakref.ref(optimizer)
    training_loop = SLCWATrainingLoop(
        model=model, triples_factory=triples_factory, optimizer=optimizer, automatic_memory_optimization=False
    )
    del optimizer

    _train(training_loop, triples_factory, clear_optimizer=True)
    assert training_loop.optimizer is None
    gc.collect()
    assert reference() is None

    with pytest.raises(ValueError, match="optimizer has been cleared"):
        _train(training_loop, triples_factory, continue_training=True)

    # a pre-instantiated optimizer cannot be re-created
    with pytest.raises(ValueError, match="pre-instantiated"):
        _train(training_loop, triples_factory)


def test_clear_optimizer_recreate(triples_factory: CoreTriplesFactory) -> None:
    """Test that a fresh run after clearing re-creates the optimizer from its hint."""
    model = TransE(triples_factory=triples_factory, embedding_dim=2)
    training_loop = SLCWATrainingLoop(
        model=model, triples_factory=triples_factory, optimizer=Adam, automatic_memory_optimization=False
    )
    _train(training_loop, triples_factory, clear_optimizer=True)
    assert training_loop.optimizer is None

    _train(training_loop, triples_factory)
    assert isinstance(training_loop.optimizer, Adam)
