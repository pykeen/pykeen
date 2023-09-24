# -*- coding: utf-8 -*-

"""Tests for initializers."""

from typing import ClassVar

import torch
from class_resolver import HintOrType

import pykeen.nn.init
from pykeen.datasets import Nations
from pykeen.nn.modules import ComplExInteraction, Interaction, QuatEInteraction
from tests import cases

from ..utils import needs_packages


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
    # quaternion needs shape to end on 4
    shape = (2, 4)
    interaction: ClassVar[HintOrType[Interaction]] = QuatEInteraction

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


@needs_packages("transformers")
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


class WeisfeilerLehmanInitializerTestCase(cases.InitializerTestCase):
    """Tests for Weisfeiler-Lehman features."""

    def setUp(self) -> None:
        """Prepare for test."""
        dataset = Nations()
        self.initializer = pykeen.nn.init.WeisfeilerLehmanInitializer(
            triples_factory=dataset.training, shape=self.shape
        )
        self.num_entities = dataset.num_entities


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

    def test_decalin(self):
        """Test decalin example."""
        # Decalin molecule from Fig 4 page 15 from the paper https://arxiv.org/pdf/2110.07875.pdf
        source, target = torch.as_tensor(
            [
                [1, 2, 3, 4, 5, 5, 0, 0, 6, 7, 8],
                [2, 3, 4, 5, 6, 0, 1, 9, 7, 8, 9],
            ]
        )
        # create triples with a dummy relation type 0
        decalin_triples = torch.stack([source, torch.zeros_like(source), target], dim=-1)
        templates = torch.as_tensor(
            data=[
                [0.0000, 0.5000, 0.0000, 0.3542, 0.0000],  # green
                [0.0000, 0.4167, 0.0000, 0.2824, 0.0000],  # red
                [0.0000, 0.4444, 0.0000, 0.3179, 0.0000],  # blue
            ],
        )
        # 0: green: 2, 3, 7, 8
        # 1: red: 1, 4, 6, 9
        # 2: blue: 0, 5
        colors = torch.as_tensor(data=[2, 1, 0, 0, 1, 2, 1, 0, 0, 1], dtype=torch.long)
        rwpe_vectors = templates[colors]
        initializer = pykeen.nn.init.RandomWalkPositionalEncodingInitializer(
            mapped_triples=decalin_triples,
            dim=5,
            # the example includes the first power
            skip_first_power=False,
        )
        assert torch.allclose(initializer.tensor, rwpe_vectors, rtol=1.0e-03)
