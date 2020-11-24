# -*- coding: utf-8 -*-

"""Unittest for the :mod:`pykeen.nn` module."""
import itertools
import unittest
from typing import Any, Iterable, MutableMapping, Optional, Sequence

import torch

from pykeen.nn import Embedding, LiteralRepresentations, RepresentationModule
from pykeen.nn.sim import kullback_leibler_similarity
from pykeen.testing.base import GenericTests
from pykeen.typing import GaussianDistribution


class RepresentationModuleTests(GenericTests[RepresentationModule]):
    """Tests for RepresentationModule."""

    batch_size: int = 3
    num: int = 5
    exp_shape: Sequence[int] = (5,)

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.instance.reset_parameters()

    def test_max_id(self):
        assert self.instance.max_id == self.num

    def test_shape(self):
        assert self.instance.shape == self.exp_shape

    def _test_forward(self, indices: Optional[torch.LongTensor]):
        """Test the forward method."""
        assert indices is None or (
            torch.is_tensor(indices)
            and indices.dtype == torch.long
            and indices.ndimension() == 1
        )
        x = self.instance(indices=indices)
        assert torch.is_tensor(x)
        assert x.dtype == torch.float32
        n = self.num if indices is None else indices.shape[0]
        assert x.shape == tuple([n, *self.instance.shape])
        self._verify_content(x=x, indices=indices)

    def _verify_content(self, x, indices):
        """Additional verification."""
        assert x.requires_grad

    def _test_indices(self) -> Iterable[torch.LongTensor]:
        return [
            torch.randint(self.num, size=(self.batch_size,)),
            torch.randperm(self.num),
            torch.randperm(self.num).repeat(2),
        ]

    def test_forward_without_indices(self):
        self._test_forward(indices=None)

    def test_forward_with_indices(self):
        for indices in self._test_indices():
            self._test_forward(indices=indices)

    def _test_in_canonical_shape(self, indices):
        name_to_shape = dict(h=1, r=2, t=3)
        for dim in itertools.chain(name_to_shape.keys(), name_to_shape.values()):
            # batch_size, d1, d2, d3, *
            x = self.instance.get_in_canonical_shape(dim=dim, indices=indices)
            assert torch.is_tensor(x)
            assert x.dtype == torch.float32
            assert x.ndimension() == 4 + len(self.exp_shape)
            exp_shape = [1, 1, 1, 1] + list(self.exp_shape)
            if isinstance(dim, str):
                dim = name_to_shape[dim]
            if indices is None:  # 1-n scoring
                exp_shape[dim] = self.num
            if indices is not None:  # batch dimension
                exp_shape[0] = indices.shape[0]
                if indices.ndimension() > 1:  # multi-target batching
                    exp_shape[dim] = indices.shape[1]
            assert x.shape == tuple(exp_shape)

    def test_get_in_canonical_shape_without_indices(self):
        self._test_in_canonical_shape(indices=None)

    def test_get_in_canonical_shape_with_indices(self):
        for indices in self._test_indices():
            self._test_in_canonical_shape(indices=indices)

    def test_get_in_canonical_shape_with_2d_indices(self):
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
