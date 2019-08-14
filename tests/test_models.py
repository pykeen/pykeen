# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import unittest
from typing import Any, ClassVar, Mapping, Optional, Type

import torch

import poem.models
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models import (
    BaseModule, ComplEx, ConvKB, DistMult, ERMLP, HolE, KG2E, NTN, ProjE, RESCAL, RotatE, SimplE,
    StructuredEmbedding, TransD, TransE, TransH, TransR, UnstructuredModel,
)
from poem.models.multimodal import MultimodalBaseModule
from tests.constants import TEST_DATA

SKIP_MODULES = {'BaseModule', 'MultimodalBaseModule'}


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


class TestERMLP(_ModelTestCase, unittest.TestCase):
    """Test the ERMLP model."""

    model_cls = ERMLP
    model_kwargs = {
        'hidden_dim': 32,
    }


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


class TestProjE(_ModelTestCase, unittest.TestCase):
    """Test the ProjE model."""

    model_cls = ProjE


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


class TestTesting(unittest.TestCase):
    """Yo dawg, I heard you like testing, so I wrote a test to test the tests so you can test while you're testing."""

    def test_testing(self):
        """Check that there's a test for all models.

        For now, this is excluding multimodel models. Not sure how to test those yet.
        """
        model_names = {
            cls.__name__
            for cls in BaseModule.__subclasses__()
        } - SKIP_MODULES

        tested_model_names = {
            value.model_cls.__name__
            for name, value in globals().items()
            if (
                isinstance(value, type)
                and issubclass(value, _ModelTestCase)
                and not name.startswith('_')
                and not issubclass(value.model_cls, MultimodalBaseModule)
            )
        } - SKIP_MODULES

        self.assertEqual(model_names, tested_model_names, msg='Some models have not been tested')

    def test_importing(self):
        """Test that all models are available from :mod:`poem.models`."""
        models_path = os.path.abspath(os.path.dirname(poem.models.__file__))

        model_names = set()
        for directory, subdirectories, filenames in os.walk(models_path):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue

                path = os.path.join(directory, filename)
                relpath = os.path.relpath(path, models_path)
                if relpath.endswith('__init__.py'):
                    continue

                import_path = 'poem.models.' + relpath[:-len('.py')].replace(os.sep, '.')
                module = importlib.import_module(import_path)

                for name in dir(module):
                    value = getattr(module, name)
                    if (
                        isinstance(value, type)
                        and issubclass(value, poem.models.BaseModule)
                    ):
                        model_names.add(value.__name__)

        star_model_names = set(poem.models.__all__) - SKIP_MODULES
        model_names -= SKIP_MODULES

        self.assertEqual(model_names, star_model_names, msg='Forgot to add some imports')

        for name in model_names:
            self.assertIn(f':py:class:`poem.models.{name}`', poem.models.__doc__, msg=f'Forgot to document {name}')
