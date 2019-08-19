# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import unittest
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from torch.optim import Adagrad

import poem.models
from poem.datasets.nations import NationsTrainingTriplesFactory
from poem.instance_creation_factories import TriplesFactory
from poem.models import (
    ComplEx, ConvKB, DistMult, ERMLP, HolE, KG2E, NTN, ProjE, RESCAL, RotatE, SimplE,
    StructuredEmbedding, TransD, TransE, TransH, TransR, UnstructuredModel,
)
from poem.models.base import BaseModule, RegularizedModel
from poem.models.multimodal import MultimodalBaseModule
from poem.training import CWATrainingLoop, OWATrainingLoop
from poem.training.cwa import CWANotImplementedError

SKIP_MODULES = {'BaseModule', 'MultimodalBaseModule', 'RegularizedModel'}


class _ModelTestCase:
    """A test case for quickly defining common tests for KGE models."""

    model_cls: ClassVar[Type[BaseModule]]
    model_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    batch_size: int
    embedding_dim: int
    factory: TriplesFactory
    model: BaseModule

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.batch_size = 16
        self.embedding_dim = 8
        self.factory = NationsTrainingTriplesFactory()
        self.model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            init=True,
            **(self.model_kwargs or {})
        )

    def test_init(self):
        """Test the model's ``init_empty_weights_()`` function."""
        # get number of parameters and shape
        init_model = self.model.init_empty_weights_()
        assert init_model == self.model
        params = list(init_model.parameters())
        param_shapes = set(p.shape for p in params)
        num_params = len(params)

        # clear model
        clear_model = self.model.clear_weights_()
        assert clear_model == self.model
        # init model
        new_model = self.model.init_empty_weights_()
        assert new_model == self.model

        # check if number and shapes match
        params = list(init_model.parameters())
        new_num_params = len(params)
        assert new_num_params == num_params
        new_param_shapes = set(p.shape for p in params)
        assert new_param_shapes == param_shapes

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        assert torch.all(torch.isfinite(scores)).item()

    def test_forward_owa(self) -> None:
        """Test the model's ``forward_owa()`` function."""
        batch = torch.zeros(self.batch_size, 3, dtype=torch.long)
        try:
            scores = self.model.forward_owa(batch)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(batch, scores)

    def test_forward_cwa(self) -> None:
        """Test the model's ``forward_cwa()`` function."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)
        try:
            scores = self.model.forward_cwa(batch)
        except NotImplementedError:
            self.fail(msg='Forward CWA not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    def test_forward_inverse_cwa(self) -> None:
        """Test the model's ``forward_inverse_cwa()`` function."""
        batch = torch.zeros(self.batch_size, 2, dtype=torch.long)
        try:
            scores = self.model.forward_inverse_cwa(batch)
        except NotImplementedError:
            self.fail(msg='Forward Inverse CWA not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    def test_train_owa(self) -> None:
        """Test that OWA training does not fail."""
        optimizer_instance = Adagrad(params=self.model.get_grad_params(), lr=0.001)
        loop = OWATrainingLoop(model=self.model, optimizer=optimizer_instance)
        try:
            losses = loop.train(num_epochs=5, batch_size=128)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        self.assertIsInstance(losses, list)

    def test_train_cwa(self) -> None:
        """Test that CWA training does not fail."""
        optimizer_instance = Adagrad(params=self.model.get_grad_params(), lr=0.001)
        try:
            loop = CWATrainingLoop(model=self.model, optimizer=optimizer_instance)
        except CWANotImplementedError as e:
            self.skipTest(str(e))

        losses = loop.train(num_epochs=5, batch_size=128)
        self.assertIsInstance(losses, list)


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
            for cls in BaseModule.__subclasses__() + RegularizedModel.__subclasses__()
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
                        and issubclass(value, BaseModule)
                    ):
                        model_names.add(value.__name__)

        star_model_names = set(poem.models.__all__) - SKIP_MODULES
        model_names -= SKIP_MODULES

        self.assertEqual(model_names, star_model_names, msg='Forgot to add some imports')

        for name in model_names:
            self.assertIn(f':py:class:`poem.models.{name}`', poem.models.__doc__, msg=f'Forgot to document {name}')
