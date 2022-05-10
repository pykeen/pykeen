# -*- coding: utf-8 -*-

"""Tests for initializers."""

import unittest

import torch

import pykeen.nn.init
from pykeen.datasets import Nations
from pykeen.nn.modules import ComplExInteraction
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
    # complex tensor
    dtype = torch.cfloat
    interaction = ComplExInteraction

    def _verify_initialization(self, x: torch.FloatTensor) -> None:  # noqa: D102
        # check value range
        assert (x >= -1.0).all()
        assert (x <= 1.0).all()
        # check modulus == 1
        mod = torch.view_as_complex(x).abs()
        assert torch.allclose(mod, torch.ones_like(mod))


class PretrainedInitializerTestCase(cases.InitializerTestCase):
    """Tests for initialization from pretrained embedding."""

    def setUp(self) -> None:
        """Prepare for test."""
        self.pretrained = torch.rand(self.num_entities, *self.shape)
        self.initializer = pykeen.nn.init.PretrainedInitializer(tensor=self.pretrained)

    def _verify_initialization(self, x: torch.FloatTensor) -> None:  # noqa: D102
        assert (x == self.pretrained).all()


class QuaternionTestCase(cases.InitializerTestCase):
    """Tests for quaternion initialization."""

    initializer = staticmethod(pykeen.nn.init.init_quaternions)
    # quaternion needs dim divisible by 4
    shape = (4,)

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
        self.num_entities = dataset.num_entities
        self.shape = self.initializer.tensor.shape[1:]


class RandomWalkPositionalEncodingInitializerTestCase(cases.InitializerTestCase):
    """Tests for random-walk positional encoding."""

    def setUp(self) -> None:
        """Prepare for test."""
        dataset = Nations()
        self.triples_factory = dataset.training
        self.initializer = pykeen.nn.init.RandomWalkPositionalEncodingInitializer(
            triples_factory=self.triples_factory, dim=3
        )
        self.num_entities = dataset.num_entities
        self.shape = self.initializer.tensor.shape[1:]

    def test_invariances(self):
        """Test some invariances."""
        self.initializer: pykeen.nn.init.PretrainedInitializer
        x = self.initializer.tensor
        # value range
        assert (x >= 0).all()
        assert (x <= 1).all()
        # highest degree node has largest value
        uniq, counts = self.triples_factory.mapped_triples[:, 0::2].unique(return_counts=True)
        center = uniq[counts.argmax()]
        assert (x.argmax(dim=0) == center).all()
