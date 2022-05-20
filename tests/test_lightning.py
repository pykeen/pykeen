"""Tests for training with PyTorch Lightning."""

import itertools

import pytest

from pykeen.contrib.lightning import lit_module_resolver, lit_pipeline
from pykeen.models import model_resolver

LIT_MODULES = lit_module_resolver.lookup_dict.keys()
INTERACTIONS = model_resolver.lookup_dict.keys()


# test combinations of models with training loops
@pytest.mark.parametrize(("model", "training_loop"), itertools.product(INTERACTIONS, LIT_MODULES))
def test_lit_training(model, training_loop):
    """Test training models with PyTorch Lightning."""
    lit_pipeline(
        training_loop=training_loop,
        training_loop_kwargs=dict(
            model=model,
            # use a small configuration for testing
            # TODO: this does not properly work for all models
            dataset="nations",
            model_kwargs=dict(embedding_dim=8),
            batch_size=8,
        ),
        trainer_kwargs=dict(
            # automatically choose accelerator
            accelerator="auto",
            # defaults to TensorBoard; explicitly disabled here
            logger=False,
            # disable checkpointing
            enable_checkpointing=False,
            # fast run
            max_epochs=2,
        ),
    )
