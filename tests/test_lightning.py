"""Tests for training with PyTorch Lightning."""

import itertools

import pytest
import pytorch_lightning

from pykeen.contrib.lightning import lit_module_resolver
from pykeen.models import model_resolver

LIT_MODULES = lit_module_resolver.lookup_dict.keys()
INTERACTIONS = model_resolver.lookup_dict.keys()


# test combinations of models with training loops
@pytest.mark.parametrize(("model", "training_loop"), itertools.product(INTERACTIONS, LIT_MODULES))
def test_lit_training(model, training_loop):
    """Test training models with PyTorch Lightning."""
    lit = lit_module_resolver.make(
        training_loop,
        model=model,
        # use a small configuration for testing
        # TODO: this does not properly work for all models
        dataset="nations",
        model_kwargs=dict(embedding_dim=8),
        batch_size=8,
    )
    trainer = pytorch_lightning.Trainer(
        # automatically choose accelerator
        accelerator="auto",
        # defaults to TensorBoard; explicitly disabled here
        logger=False,
        # disable checkpointing
        enable_checkpointing=False,
        # fast run
        max_epochs=2,
    )
    trainer.fit(model=lit)
