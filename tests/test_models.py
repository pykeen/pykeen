# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import unittest
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from torch.optim import Adagrad

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import BaseModule
from poem.models.unimodal import (
    ComplEx, ConvKB, DistMult, HolE, NTN, RESCAL, RotatE, StructuredEmbedding, TransD,
    TransE, TransH, TransR, UnstructuredModel,
)
from poem.training import CWATrainingLoop, OWATrainingLoop
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
        self.model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            **(self.model_kwargs or {}),
        )

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""

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

    def test_train_owa(self) -> None:
        """Test that OWA training does not fail."""
        optimizer_instance = Adagrad(params=self.model.get_grad_params(), lr=0.001)
        loop = OWATrainingLoop(model=self.model, optimizer=optimizer_instance)
        loop.train(num_epochs=5, batch_size=128)

    def test_train_cwa(self) -> None:
        """Test that CWA training does not fail."""
        optimizer_instance = Adagrad(params=self.model.get_grad_params(), lr=0.001)
        loop = CWATrainingLoop(model=self.model, optimizer=optimizer_instance)
        loop.train(num_epochs=5, batch_size=128)


class _DistanceModelTestCase(_ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class TestComplex(_ModelTestCase, unittest.TestCase):
    """Test the ComplEx model."""

    model_cls = ComplEx


@skip_until_fixed
class TestConvKB(_ModelTestCase, unittest.TestCase):
    """Test the ConvKB model."""

    model_cls = ConvKB


class TestDistMult(_ModelTestCase, unittest.TestCase):
    """Test the DistMult model."""

    model_cls = DistMult


class TestHolE(_ModelTestCase, unittest.TestCase):
    """Test the HolE model."""

    model_cls = HolE


class TestNTN(_ModelTestCase, unittest.TestCase):
    """Test the NTN model."""

    model_cls = NTN


@skip_until_fixed
class TestRESCAL(_ModelTestCase, unittest.TestCase):
    """Test the RESCAL model."""

    model_cls = RESCAL


class TestRotatE(_ModelTestCase, unittest.TestCase):
    """Test the RotatE model."""

    model_cls = RotatE


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
