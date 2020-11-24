# -*- coding: utf-8 -*-

"""Unittest for the :mod:`pykeen.nn` module."""
import itertools
import unittest
from typing import Any, Iterable, MutableMapping, Optional, Sequence
from unittest.mock import Mock

import numpy
import pytest
import torch

from pykeen.nn import Embedding, EmbeddingSpecification, LiteralRepresentations, RepresentationModule
from pykeen.nn.representation import DIMS, get_expected_canonical_shape
from pykeen.nn.sim import kullback_leibler_similarity
from pykeen.testing.base import GenericTests, TestsTest
from pykeen.typing import GaussianDistribution


class RepresentationModuleTests(GenericTests[RepresentationModule]):
    """Tests for RepresentationModule."""

    #: The batch size
    batch_size: int = 3

    #: The number of representations
    num: int = 5

    #: The expected shape of an individual representation
    exp_shape: Sequence[int] = (5,)

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.instance.reset_parameters()

    def test_max_id(self):
        """Test the maximum ID."""
        assert self.instance.max_id == self.num

    def test_shape(self):
        """Test the shape."""
        assert self.instance.shape == self.exp_shape

    def _test_forward(self, indices: Optional[torch.LongTensor]):
        """Test the forward method."""
        x = self.instance(indices=indices)
        assert torch.is_tensor(x)
        assert x.dtype == torch.float32
        n = self.num if indices is None else indices.shape[0]
        assert x.shape == tuple([n, *self.instance.shape])
        self._verify_content(x=x, indices=indices)

    def _verify_content(self, x, indices):
        """Additional verification."""
        assert x.requires_grad

    def _valid_indices(self) -> Iterable[torch.LongTensor]:
        return [
            torch.randint(self.num, size=(self.batch_size,)),
            torch.randperm(self.num),
            torch.randperm(self.num).repeat(2),
        ]

    def _invalid_indices(self) -> Iterable[torch.LongTensor]:
        return [
            torch.as_tensor([self.num], dtype=torch.long),  # too high index
            torch.randint(self.num, size=(2, 3)),  # too many indices
        ]

    def test_forward_without_indices(self):
        """Test forward without providing indices."""
        self._test_forward(indices=None)

    def test_forward_with_indices(self):
        """Test forward with providing indices."""
        for indices in self._valid_indices():
            self._test_forward(indices=indices)

    def test_forward_with_invalid_indices(self):
        """Test whether passing invalid indices crashes."""
        for indices in self._invalid_indices():
            with pytest.raises((IndexError, RuntimeError)):
                self._test_forward(indices=indices)

    def _test_in_canonical_shape(self, indices: Optional[torch.LongTensor]):
        """Test get_in_canonical_shape with the given indices."""
        # test both, using the actual dimension, and its name
        for dim in itertools.chain(DIMS.keys(), DIMS.values()):
            # batch_size, d1, d2, d3, *
            x = self.instance.get_in_canonical_shape(dim=dim, indices=indices)

            # data type
            assert torch.is_tensor(x)
            assert x.dtype == torch.float32  # todo: adjust?
            assert x.ndimension() == 4 + len(self.exp_shape)

            # get expected shape
            exp_shape = get_expected_canonical_shape(
                indices=indices,
                dim=dim,
                suffix_shape=self.exp_shape,
                num=self.num,
            )
            assert x.shape == exp_shape

    def test_get_in_canonical_shape_without_indices(self):
        """Test get_in_canonical_shape without indices, i.e. with 1-n scoring."""
        self._test_in_canonical_shape(indices=None)

    def test_get_in_canonical_shape_with_indices(self):
        """Test get_in_canonical_shape with 1-dimensional indices."""
        for indices in self._valid_indices():
            self._test_in_canonical_shape(indices=indices)

    def test_get_in_canonical_shape_with_2d_indices(self):
        """Test get_in_canonical_shape with 2-dimensional indices."""
        indices = torch.randint(self.num, size=(self.batch_size, 2))
        self._test_in_canonical_shape(indices=indices)


class EmbeddingTests(RepresentationModuleTests, unittest.TestCase):
    """Tests for Embedding."""

    cls = Embedding
    kwargs = dict(
        num_embeddings=RepresentationModuleTests.num,
        shape=RepresentationModuleTests.exp_shape,
    )


class TensorEmbeddingTests(RepresentationModuleTests, unittest.TestCase):
    """Tests for Embedding with 2-dimensional shape."""

    cls = Embedding
    exp_shape = (3, 7)
    kwargs = dict(
        num_embeddings=RepresentationModuleTests.num,
        shape=(3, 7),
    )


class LiteralRepresentationsTests(RepresentationModuleTests, unittest.TestCase):
    """Tests for literal embeddings."""

    cls = LiteralRepresentations

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        self.numeric_literals = torch.rand(self.num, *self.exp_shape)
        kwargs["numeric_literals"] = self.numeric_literals
        return kwargs

    def _verify_content(self, x, indices):  # noqa: D102
        exp_x = self.numeric_literals
        if indices is not None:
            exp_x = exp_x[indices]
        assert torch.allclose(x, exp_x)


class RepresentationModuleTestsTest(TestsTest[RepresentationModule], unittest.TestCase):
    """Test that there are tests for all representation modules."""

    base_cls = RepresentationModule
    base_test = RepresentationModuleTests


class EmbeddingSpecificationTests(unittest.TestCase):
    """Tests for EmbeddingSpecification."""

    #: The number of embeddings
    num: int = 3

    def test_make(self):
        """Test make."""
        initializer = Mock()
        normalizer = Mock()
        constrainer = Mock()
        regularizer = Mock()
        for embedding_dim, shape in [
            (None, (3,)),
            (None, (3, 5)),
            (3, None),
        ]:
            spec = EmbeddingSpecification(
                embedding_dim=embedding_dim,
                shape=shape,
                initializer=initializer,
                normalizer=normalizer,
                constrainer=constrainer,
                regularizer=regularizer,
            )
            emb = spec.make(num_embeddings=self.num)

            # check shape
            assert emb.embedding_dim == (embedding_dim or int(numpy.prod(shape)))
            assert emb.shape == (shape or (embedding_dim,))
            assert emb.num_embeddings == self.num

            # check attributes
            assert emb.initializer is initializer
            assert emb.normalizer is normalizer
            assert emb.constrainer is constrainer
            assert emb.regularizer is regularizer


class KullbackLeiblerTests(unittest.TestCase):
    """Tests for the vectorized computation of KL divergences."""

    d: int = 3

    def setUp(self) -> None:  # noqa: D102
        self.e_mean = torch.rand(self.d)
        self.e_var = torch.rand(self.d).exp()
        self.r_mean = torch.rand(self.d)
        self.r_var = torch.rand(self.d).exp()

    def _get_e(self, pre_shape=(1, 1, 1)):
        return GaussianDistribution(
            mean=self.e_mean.view(*pre_shape, self.d),
            diagonal_covariance=self.e_var.view(*pre_shape, self.d),
        )

    def _get_r(self, pre_shape=(1, 1)):
        return GaussianDistribution(
            mean=self.r_mean.view(*pre_shape, self.d),
            diagonal_covariance=self.r_var.view(*pre_shape, self.d),
        )

    def test_against_torch_builtin(self):
        """Compare value against torch.distributions."""
        # r: (batch_size, num_heads, num_tails, d)
        e = self._get_e()
        # r: (batch_size, num_relations, d)
        r = self._get_r()
        sim = kullback_leibler_similarity(e=e, r=r, exact=True).view(-1)

        p = torch.distributions.MultivariateNormal(loc=self.e_mean, covariance_matrix=torch.diag(self.e_var))
        q = torch.distributions.MultivariateNormal(loc=self.r_mean, covariance_matrix=torch.diag(self.r_var))
        sim2 = -torch.distributions.kl_divergence(p=p, q=q).view(-1)
        assert torch.allclose(sim, sim2)

    def test_self_similarity(self):
        """Check value of similarity to self."""
        # e: (batch_size, num_heads, num_tails, d)
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence = 0 => similarity = -divergence = 0
        e = self._get_e()
        r = self._get_e(pre_shape=(1, 1))
        sim = kullback_leibler_similarity(e=e, r=r, exact=True)
        assert torch.allclose(sim, torch.zeros_like(sim))

    def test_value_range(self):
        """Check the value range."""
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence >= 0 => similarity = -divergence <= 0
        e = self._get_e()
        r = self._get_r()
        sim = kullback_leibler_similarity(e=e, r=r, exact=True)
        assert (sim <= 0).all()
