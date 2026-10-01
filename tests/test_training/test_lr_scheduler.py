"""Tests for LR schedulers."""

import pathlib
from hashlib import md5

import pytest
import torch
from class_resolver import HintOrType, OptionalKwargs
from torch.optim import lr_scheduler

from pykeen.datasets import Nations
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.training import SLCWATrainingLoop
from pykeen.training.training_loop import CheckpointMismatchError
from pykeen.triples import CoreTriplesFactory


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [(None, None), ("CosineAnnealingWarmRestarts", {"T_0": 10})],
)
def test_lr_scheduler(cls: HintOrType[lr_scheduler.LRScheduler], kwargs: OptionalKwargs) -> None:
    """Smoke-test for training with learning rate schedule."""
    pipeline(
        dataset="nations",
        model="mure",
        model_kwargs={"embedding_dim": 2},
        training_kwargs={"num_epochs": 1},
        lr_scheduler=cls,
        lr_scheduler_kwargs=kwargs,
    )


NUM_EPOCHS = 4
CHECKPOINT_NAME = "checkpoint.pt"


@pytest.fixture(scope="module")
def triples_factory() -> CoreTriplesFactory:
    """Return the training triples factory."""
    return Nations().training


def _make_training_loop(triples_factory: CoreTriplesFactory, lr: float = 0.1) -> SLCWATrainingLoop:
    model = TransE(triples_factory=triples_factory, embedding_dim=8, random_seed=0)
    return SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        lr_scheduler="ExponentialLR",
        lr_scheduler_kwargs={"gamma": 0.5},
        automatic_memory_optimization=False,
    )


def _train(
    training_loop: SLCWATrainingLoop, triples_factory: CoreTriplesFactory, directory: pathlib.Path, num_epochs: int
) -> list[float]:
    return training_loop.train(
        triples_factory=triples_factory,
        num_epochs=num_epochs,
        batch_size=128,
        checkpoint_name=CHECKPOINT_NAME,
        checkpoint_directory=directory,
        checkpoint_frequency=0,
        use_tqdm=False,
    )


def test_resume_from_checkpoint(triples_factory: CoreTriplesFactory, tmp_path: pathlib.Path) -> None:
    """Test that training with an LR scheduler can be resumed from a checkpoint."""
    losses = _make_training_loop(triples_factory).train(
        triples_factory=triples_factory, num_epochs=NUM_EPOCHS, batch_size=128, use_tqdm=False
    )
    _train(_make_training_loop(triples_factory), triples_factory, tmp_path, num_epochs=NUM_EPOCHS // 2)
    losses_resumed = _train(_make_training_loop(triples_factory), triples_factory, tmp_path, num_epochs=NUM_EPOCHS)
    assert losses == pytest.approx(losses_resumed)


def test_resume_from_checkpoint_mismatch(triples_factory: CoreTriplesFactory, tmp_path: pathlib.Path) -> None:
    """Test that resuming from a checkpoint with a different optimizer configuration fails."""
    _train(_make_training_loop(triples_factory), triples_factory, tmp_path, num_epochs=NUM_EPOCHS // 2)
    with pytest.raises(CheckpointMismatchError):
        _train(_make_training_loop(triples_factory, lr=0.01), triples_factory, tmp_path, num_epochs=NUM_EPOCHS)


def test_resume_from_legacy_checkpoint(triples_factory: CoreTriplesFactory, tmp_path: pathlib.Path) -> None:
    """Test that training can be resumed from a checkpoint with the legacy checksum."""
    training_loop = _make_training_loop(triples_factory)
    _train(training_loop, triples_factory, tmp_path, num_epochs=NUM_EPOCHS // 2)
    # rewrite the checkpoint to the legacy format, whose checksum was calculated from the string representations
    path = tmp_path.joinpath(CHECKPOINT_NAME)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["checksum_version"]
    h = md5()  # noqa: S324
    h.update(str(training_loop.model).encode("utf-8"))
    h.update(str(training_loop.optimizer).encode("utf-8"))
    checkpoint["checksum"] = h.hexdigest()
    torch.save(checkpoint, path)

    _train(_make_training_loop(triples_factory), triples_factory, tmp_path, num_epochs=NUM_EPOCHS)
