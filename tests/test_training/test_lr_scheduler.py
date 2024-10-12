"""Tests for LR schedulers."""

import pytest
from class_resolver import HintOrType, OptionalKwargs
from torch.optim import lr_scheduler

from pykeen.pipeline import pipeline


@pytest.mark.parametrize(
    "cls, kwargs",
    [(None, None), ("CosineAnnealingWarmRestarts", {"T_0": 10})],
)
def test_lr_scheduler(cls: HintOrType[lr_scheduler.LRScheduler], kwargs: OptionalKwargs) -> None:
    """Smoke-test for training with learning rate schedule."""
    pipeline(
        dataset="nations",
        model="mure",
        model_kwargs=dict(embedding_dim=2),
        training_kwargs=dict(num_epochs=1),
        lr_scheduler=cls,
        lr_scheduler_kwargs=kwargs,
    )
