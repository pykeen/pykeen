# -*- coding: utf-8 -*-

"""Unittest for the :mod:`pykeen.nn` module."""

import unittest

import torch

from pykeen.nn import Embedding
from pykeen.nn.sim import kullback_leibler_similarity
from pykeen.typing import GaussianDistribution


class EmbeddingsInCanonicalShapeTests(unittest.TestCase):
    """Test get_embedding_in_canonical_shape()."""

    #: The number of embeddings
    num_embeddings: int = 3

    #: The embedding dimension
    embedding_dim: int = 2

    def setUp(self) -> None:
        """Initialize embedding."""
        self.embedding = Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.generator = torch.manual_seed(42)
        self.embedding._embeddings.weight.data = torch.rand(
            self.num_embeddings,
            self.embedding_dim,
            generator=self.generator,
        )

    def test_no_indices(self):
        """Test getting all embeddings."""
        emb = self.embedding.get_in_canonical_shape(indices=None)

        # check shape
        assert emb.shape == (1, self.num_embeddings, self.embedding_dim)

        # check values
        exp = self.embedding(indices=None).view(1, self.num_embeddings, self.embedding_dim)
        assert torch.allclose(emb, exp)

    def _test_with_indices(self, indices: torch.Tensor) -> None:
        """Help tests with index."""
        emb = self.embedding.get_in_canonical_shape(indices=indices)

        # check shape
        num_ind = indices.shape[0]
        assert emb.shape == (num_ind, 1, self.embedding_dim)

        # check values
        exp = torch.stack([self.embedding(i) for i in indices], dim=0).view(num_ind, 1, self.embedding_dim)
        assert torch.allclose(emb, exp)

    def test_with_consecutive_indices(self):
        """Test to retrieve all embeddings with consecutive indices."""
        indices = torch.arange(self.num_embeddings, dtype=torch.long)
        self._test_with_indices(indices=indices)

    def test_with_indices_with_duplicates(self):
        """Test to retrieve embeddings at random positions with duplicate indices."""
        indices = torch.randint(
            self.num_embeddings,
            size=(2 * self.num_embeddings,),
            dtype=torch.long,
            generator=self.generator,
        )
        self._test_with_indices(indices=indices)


class KullbackLeiblerTests(unittest.TestCase):
    """Tests for the vectorized computation of KL divergences."""

    d: int = 3

    def setUp(self) -> None:
        self.e_mean = torch.rand(self.d)
        self.e_var = torch.rand(self.d).exp()
        self.r_mean = torch.rand(self.d)
        self.r_var = torch.rand(self.d).exp()

    def get_e(self, pre_shape=(1, 1, 1)):
        return GaussianDistribution(
            mean=self.e_mean.view(*pre_shape, self.d),
            diagonal_covariance=self.e_var.view(*pre_shape, self.d),
        )

    def get_r(self, pre_shape=(1, 1)):
        return GaussianDistribution(
            mean=self.r_mean.view(*pre_shape, self.d),
            diagonal_covariance=self.r_var.view(*pre_shape, self.d),
        )

    def test_against_torch_builtin(self):
        """Compare value against torch.distributions."""
        # r: (batch_size, num_heads, num_tails, d)
        e = self.get_e()
        # r: (batch_size, num_relations, d)
        r = self.get_r()
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
        e = self.get_e()
        r = self.get_e(pre_shape=(1, 1))
        sim = kullback_leibler_similarity(e=e, r=r, exact=True)
        assert torch.allclose(sim, torch.zeros_like(sim))

    def test_value_range(self):
        """Check the value range."""
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence >= 0 => similarity = -divergence <= 0
        e = self.get_e()
        r = self.get_r()
        sim = kullback_leibler_similarity(e=e, r=r, exact=True)
        assert (sim <= 0).all()
