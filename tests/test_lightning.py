"""Tests for training with PyTorch Lightning."""
import unittest
from typing import Any, ClassVar, Mapping

import unittest_templates

try:
    import pytorch_lightning

    from pykeen.contrib.lightning import LCWALitModule, LitModule, SLCWALitModule
except ImportError:
    pytorch_lightning = None
    LitModule = LCWALitModule = SLCWALitModule = None


@unittest.skipIf(
    pytorch_lightning is None, reason="PyTorch Lightning tests require `pytorch-lightning` to be installed"
)
class LitModuleTestCase(unittest_templates.GenericTestCase[LitModule]):
    """A base test for PyTorch Lightning training."""

    kwargs: ClassVar[Mapping[str, Any]] = dict(
        # dataset
        dataset="nations",
        dataset_kwargs=None,
        mode=None,
        # model
        model="distmult",  # TODO: we may want to train more than one interaction function
        # choose a small embedding dim for fast tests
        model_kwargs=dict(embedding_dim=16),
        batch_size=8,
        learning_rate=1.0e-03,
        label_smoothing=0.0,
        # optimizer
        optimizer="adam",
        optimizer_kwargs=None,
    )

    def test_training(self):
        """Test training."""
        pytorch_lightning.Trainer(
            # automatically choose accelerator
            accelerator="auto",
            # defaults to TensorBoard; explicitly disabled here
            logger=False,
            # disable checkpointing
            enable_checkpointing=False,
            # fast run
            max_epochs=2,
        ).fit(model=self.instance)


class LCWALitModuleTestCase(LitModuleTestCase):
    """Tests for training with LCWA."""

    cls = LCWALitModule


class SLCWALitModuleTestCase(LitModuleTestCase):
    """Tests for training with LCWA."""

    cls = SLCWALitModule
