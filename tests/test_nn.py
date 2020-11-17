# -*- coding: utf-8 -*-

"""Unittest for the :mod:`pykeen.nn` module."""

import unittest

import torch

from pykeen.nn import Embedding


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
