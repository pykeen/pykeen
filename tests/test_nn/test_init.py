"""Tests for initializers."""
import torch

import pykeen.nn.init
from tests import cases


class PretrainedInitializerTestCase(cases.InitializerTestCase):
    """Tests for initialization from pretrained embedding."""

    def setUp(self) -> None:
        """Prepare for test."""
        self.pretrained = torch.rand(*self.shape)
        self.initializer = pykeen.nn.init.create_init_from_pretrained(pretrained=self.pretrained)

    def _verify_initialization(self, x: torch.FloatTensor) -> None:  # noqa: D102
        assert (x == self.pretrained).all()
