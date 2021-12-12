# -*- coding: utf-8 -*-

"""Tests for initializers."""

import unittest

import torch

import pykeen.nn.init
from pykeen.datasets import Nations
from tests import cases

try:
    import transformers
except ImportError:
    transformers = None


class NormalizationMixin:
    """Mixin for verification of unit length."""

    def _verify_initialization(self, x: torch.FloatTensor) -> None:  # noqa: D102
        xn = x.norm(dim=-1)
        assert torch.allclose(xn, torch.ones_like(xn))


class NormalNormTestCase(NormalizationMixin, cases.InitializerTestCase):
    """Tests for normal initialization + normalization."""

    initializer = staticmethod(pykeen.nn.init.normal_norm_)


class PhasesTestCase(cases.InitializerTestCase):
    """Tests for phase initialization."""

    initializer = staticmethod(pykeen.nn.init.init_phases)

    def _verify_initialization(self, x: torch.FloatTensor) -> None:  # noqa: D102
        # check value range
        assert (x >= -1.0).all()
        assert (x <= 1.0).all()
        # check sin**2 + cos**2 == 1
        assert torch.allclose(x.view(*x.shape[:-1], 2, -1).pow(2).sum(dim=-2), torch.ones(*x.shape[:-1], 1))


class PretrainedInitializerTestCase(cases.InitializerTestCase):
    """Tests for initialization from pretrained embedding."""

    def setUp(self) -> None:
        """Prepare for test."""
        self.pretrained = torch.rand(*self.shape)
        self.initializer = pykeen.nn.init.PretrainedInitializer(tensor=self.pretrained)

    def _verify_initialization(self, x: torch.FloatTensor) -> None:  # noqa: D102
        assert (x == self.pretrained).all()


class QuaternionTestCase(cases.InitializerTestCase):
    """Tests for quaternion initialization."""

    initializer = staticmethod(pykeen.nn.init.init_quaternions)

    def _verify_initialization(self, x: torch.FloatTensor) -> None:
        # check value range (actually [-s, +s] with s = 1/sqrt(2*n))
        assert (x >= -1.0).all()
        assert (x <= 1.0).all()


class UniformNormTestCase(NormalizationMixin, cases.InitializerTestCase):
    """Tests for uniform initialization + normalization."""

    initializer = staticmethod(pykeen.nn.init.uniform_norm_)


class XavierNormalTestCase(cases.InitializerTestCase):
    """Tests for Xavier Glorot normal initialization."""

    initializer = staticmethod(pykeen.nn.init.xavier_normal_)


class XavierNormalNormTestCase(NormalizationMixin, cases.InitializerTestCase):
    """Tests for Xavier Glorot normal initialization + normalization."""

    initializer = staticmethod(pykeen.nn.init.xavier_normal_norm_)


class XavierUniformTestCase(cases.InitializerTestCase):
    """Tests for Xavier Glorot uniform initialization."""

    initializer = staticmethod(pykeen.nn.init.xavier_uniform_)


class XavierUniformNormTestCase(NormalizationMixin, cases.InitializerTestCase):
    """Tests for Xavier Glorot uniform initialization + normalization."""

    initializer = staticmethod(pykeen.nn.init.xavier_uniform_norm_)


@unittest.skipIf(transformers is None, "Need to install `transformers`")
class LabelBasedInitializerTestCase(cases.InitializerTestCase):
    """Tests for label-based initialization."""

    def setUp(self) -> None:
        """Prepare for test."""
        dataset = Nations()
        self.initializer = pykeen.nn.init.LabelBasedInitializer.from_triples_factory(
            triples_factory=dataset.training,
            for_entities=True,
        )
        self.shape = self.initializer.tensor.shape
