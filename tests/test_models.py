# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import unittest
from typing import Any, ClassVar, Mapping, Optional, Type

import torch

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import BaseModule
from poem.models.unimodal import (
    ComplEx, ConvKB, DistMult, HolE, KG2E, NTN, RESCAL, RotatE, SimplE, StructuredEmbedding, TransD,
    TransE, TransH, TransR, UnstructuredModel,
)
from tests.constants import TEST_DATA

skip_until_fixed = unittest.skip('Something wrong with model. Needs fixing')


class _ModelTestCase:
    """A test case for quickly defining common tests for KGE models."""

    model_cls: ClassVar[Type[BaseModule]]
    model_kwargs: Optional[Mapping[str, Any]] = None

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.batch_size = 16
        self.embedding_dim = 8
        self.factory = TriplesFactory(path=TEST_DATA)
        self.model = self.model_cls(self.factory, embedding_dim=self.embedding_dim, **(self.model_kwargs or {}))

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        assert torch.all(torch.isfinite(scores)).item()

    def test_forward_owa(self) -> None:
        """Test the model's ``forward_owa()`` function."""
        batch = torch.zeros(self.batch_size, 3, dtype=torch.long)
        scores = self.model.forward_owa(batch)
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(batch, scores)

    def test_forward_cwa(self) -> None:
        """Test the model's ``forward_cwa()`` function."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)
        try:
            scores = self.model.forward_cwa(batch)
        except NotImplementedError:
            self.fail(msg='Forward CWA not yet implemented')
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    def test_forward_inverse_cwa(self) -> None:
        """Test the model's ``forward_inverse_cwa()`` function."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)
        try:
            scores = self.model.forward_inverse_cwa(batch)
        except NotImplementedError:
            self.fail(msg='Forward Inverse CWA not yet implemented')
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)


class _DistanceModelTestCase(_ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class TestComplex(_ModelTestCase, unittest.TestCase):
    """Test the ComplEx model."""

    model_cls = ComplEx


class TestConvKB(_ModelTestCase, unittest.TestCase):
    """Test the ConvKB model."""

    model_cls = ConvKB
    model_kwargs = {
        'num_filters': 32,
    }


class TestDistMult(_ModelTestCase, unittest.TestCase):
    """Test the DistMult model."""

    model_cls = DistMult


class TestHolE(_ModelTestCase, unittest.TestCase):
    """Test the HolE model."""

    model_cls = HolE


class TestCaseKG2EWithKL(_ModelTestCase, unittest.TestCase):
    """Test the KG2E model with KL similarity."""

    model_cls = KG2E
    model_kwargs = {
        'dist_similarity': 'KL',
    }


class TestCaseKG2EWithEL(_ModelTestCase, unittest.TestCase):
    """Test the KG2E model with EL similarity."""

    model_cls = KG2E
    model_kwargs = {
        'dist_similarity': 'EL',
    }


class TestNTN(_ModelTestCase, unittest.TestCase):
    """Test the NTN model."""

    model_cls = NTN


class TestRESCAL(_ModelTestCase, unittest.TestCase):
    """Test the RESCAL model."""

    model_cls = RESCAL


class TestRotatE(_ModelTestCase, unittest.TestCase):
    """Test the RotatE model."""

    model_cls = RotatE


class TestSimplE(_ModelTestCase, unittest.TestCase):
    """Test the SimplE model."""

    model_cls = SimplE


class TestSE(_ModelTestCase, unittest.TestCase):
    """Test the Structured Embedding model."""

    model_cls = StructuredEmbedding


class TestTransD(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransD model."""

    model_cls = TransD


class TestTransE(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransE model."""

    model_cls = TransE


class TestTransH(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransH model."""

    model_cls = TransH


class TestTransR(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransR model."""

    model_cls = TransR


class TestUM(_DistanceModelTestCase, unittest.TestCase):
    """Test the Unstructured Model."""

    model_cls = UnstructuredModel
