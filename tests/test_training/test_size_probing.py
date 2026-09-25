"""Tests for the state handling of size probing."""

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from pykeen.datasets import Nations
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import CoreTriplesFactory


@pytest.fixture
def triples_factory() -> CoreTriplesFactory:
    """Return the training triples of a small dataset."""
    return Nations().training


@pytest.fixture
def training_loop(triples_factory: CoreTriplesFactory) -> SLCWATrainingLoop:
    """Return a training loop with a stateful optimizer and an LR scheduler."""
    model = TransE(triples_factory=triples_factory, embedding_dim=2, random_seed=0)
    return SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer=Adam,
        optimizer_kwargs={"lr": 0.1},
        lr_scheduler=StepLR,
        lr_scheduler_kwargs={"step_size": 1, "gamma": 0.5},
    )


def _snapshot(training_loop: SLCWATrainingLoop) -> tuple[dict[str, torch.Tensor], int, dict]:
    assert training_loop.optimizer is not None
    assert training_loop.lr_scheduler is not None
    return (
        {key: value.detach().clone() for key, value in training_loop.model.state_dict().items()},
        len(training_loop.optimizer.state),
        training_loop.lr_scheduler.state_dict(),
    )


def _assert_restored(training_loop: SLCWATrainingLoop, snapshot) -> None:
    model_state, num_optimizer_states, lr_scheduler_state = snapshot
    for key, value in training_loop.model.state_dict().items():
        assert torch.equal(value, model_state[key]), key
    assert training_loop.optimizer is not None
    assert len(training_loop.optimizer.state) == num_optimizer_states
    assert training_loop.optimizer.param_groups[0]["lr"] == 0.1
    assert training_loop.lr_scheduler is not None
    assert training_loop.lr_scheduler.state_dict() == lr_scheduler_state


def test_size_probing(training_loop: SLCWATrainingLoop, triples_factory: CoreTriplesFactory) -> None:
    """Test that size probing applies optimizer steps, but restores the state afterwards."""
    optimizer = training_loop.optimizer
    assert optimizer is not None
    num_states_after_step: list[int] = []
    optimizer.register_step_post_hook(lambda opt, args, kwargs: num_states_after_step.append(len(opt.state)))
    snapshot = _snapshot(training_loop)

    batch_size, _ = training_loop.batch_size_search(triples_factory=triples_factory)
    training_loop.sub_batch_and_slice(batch_size=batch_size, sampler=None, triples_factory=triples_factory)

    # the optimizer state was allocated during probing ...
    assert num_states_after_step
    assert all(num_states > 0 for num_states in num_states_after_step)
    # ... and everything is restored afterwards
    assert training_loop.optimizer is optimizer
    _assert_restored(training_loop, snapshot)


def test_size_probing_error(training_loop: SLCWATrainingLoop, triples_factory: CoreTriplesFactory) -> None:
    """Test that the state is restored if size probing fails after a parameter update."""
    assert training_loop.optimizer is not None

    def _fail(*_args, **_kwargs):
        raise RuntimeError("non-OOM failure")

    training_loop.optimizer.register_step_post_hook(_fail)
    snapshot = _snapshot(training_loop)

    with pytest.raises(RuntimeError, match="non-OOM failure"):
        training_loop.batch_size_search(triples_factory=triples_factory)
    _assert_restored(training_loop, snapshot)
