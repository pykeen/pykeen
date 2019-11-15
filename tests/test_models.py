# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import traceback
import unittest
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from click.testing import CliRunner, Result
from torch.optim import SGD
from torch.optim.adagrad import Adagrad

import poem.experiments
import poem.models
from poem.datasets.kinship import TRAIN_PATH as KINSHIP_TRAIN_PATH
from poem.datasets.nations import (
    NationsTrainingTriplesFactory, TEST_PATH as NATIONS_TEST_PATH,
    TRAIN_PATH as NATIONS_TRAIN_PATH,
)
from poem.models.base import BaseModule
from poem.models.cli import build_cli_from_cls
from poem.models.multimodal import MultimodalBaseModule
from poem.training import CWATrainingLoop, OWATrainingLoop, TrainingLoop
from poem.triples import TriplesFactory

SKIP_MODULES = {'BaseModule', 'MultimodalBaseModule', 'MockModel', 'models', 'get_model_cls'}


class _ModelTestCase:
    """A test case for quickly defining common tests for KGE models."""

    #: The class of the model to test
    model_cls: ClassVar[Type[BaseModule]]

    #: Additional arguments passed to the model's constructor method
    model_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    #: The triples factory instance
    factory: TriplesFactory

    #: The model instance
    model: BaseModule

    #: The batch size for use for forward_* tests
    batch_size: int = 20

    #: The embedding dimensionality
    embedding_dim: int = 3

    #: Whether to create inverse triples (needed e.g. by ConvE)
    create_inverse_triples: bool = False

    #: The sampler to use for OWA (different e.g. for R-GCN)
    sampler = 'default'

    #: The batch size for use when testing training procedures
    train_batch_size = 400

    #: The number of epochs to train the model
    train_num_epochs = 2

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.factory = NationsTrainingTriplesFactory(create_inverse_triples=self.create_inverse_triples)
        self.model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            **(self.model_kwargs or {})
        ).to_device_()

    def test_get_grad_parameters(self):
        """Test the model's ``get_grad_params()`` method."""
        # assert there is at least one trainable parameter
        assert len(list(self.model.get_grad_params())) > 0

        # Check that all the parameters actually require a gradient
        for parameter in self.model.get_grad_params():
            assert parameter.requires_grad

        # Try to initialize an optimizer
        optimizer = SGD(params=self.model.get_grad_params(), lr=1.0)
        assert optimizer is not None

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
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.model.device)
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
        batch = self.factory.mapped_triples[:self.batch_size, :2].to(self.model.device)
        # assert batch comprises (subject, relation) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
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
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.model.device)
        # assert batch comprises (relation, object) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
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
        loop = OWATrainingLoop(
            model=self.model,
            optimizer=Adagrad(params=self.model.get_grad_params(), lr=0.001),
        )
        losses = self._safe_train_loop(
            loop,
            num_epochs=self.train_num_epochs,
            batch_size=self.train_batch_size,
            sampler=self.sampler,
        )
        self.assertIsInstance(losses, list)

    def test_train_cwa(self) -> None:
        """Test that CWA training does not fail."""
        loop = CWATrainingLoop(
            model=self.model,
            optimizer=Adagrad(params=self.model.get_grad_params(), lr=0.001),
        )
        losses = self._safe_train_loop(
            loop,
            num_epochs=self.train_num_epochs,
            batch_size=self.train_batch_size,
            sampler='default',
        )
        self.assertIsInstance(losses, list)

    def _safe_train_loop(self, loop: TrainingLoop, num_epochs, batch_size, sampler):
        try:
            losses = loop.train(num_epochs=num_epochs, batch_size=batch_size, sampler=sampler)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        else:
            return losses

    @property
    def cli_extras(self):
        kwargs = self.model_kwargs or {}
        extras = []
        for k, v in kwargs.items():
            extras.append('--' + k.replace('_', '-'))
            extras.append(str(v))
        extras += [
            '--number-epochs', self.train_num_epochs,
            '--embedding-dim', self.embedding_dim,
            '--batch-size', self.train_batch_size
        ]
        extras = [str(e) for e in extras]
        return extras

    def test_cli_training_nations(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH] + self.cli_extras)

    def test_cli_training_kinship(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', KINSHIP_TRAIN_PATH] + self.cli_extras)

    def test_cli_training_nations_testing(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH, '-q', NATIONS_TEST_PATH] + self.cli_extras)

    def _help_test_cli(self, args):
        """Test running the pipeline on all models."""
        if self.model_cls is poem.models.RGCN:
            self.skipTest('R-GCN takes too long')
            # TODO: Once post_parameter_update is available, implement enrichment precomputation and remove this point.
        runner = CliRunner()
        cli = build_cli_from_cls(self.model_cls)
        # TODO: Catch HolE MKL error?
        result: Result = runner.invoke(cli, args)

        self.assertEqual(
            0,
            result.exit_code,
            msg=f'''
Command
=======
$ poem train {self.model_cls.__name__.lower()} {' '.join(args)}

Output
======
{result.output}

Exception
=========
{result.exc_info[1]}

Traceback
=========
{''.join(traceback.format_tb(result.exc_info[2]))}
            ''',
        )

    def test_has_hpo_defaults(self):
        """Test that there are defaults for HPO."""
        try:
            d = self.model_cls.hpo_default
        except AttributeError:
            self.fail(msg=f'{self.model_cls.__name__} is missing hpo_default class attribute')
        else:
            self.assertIsInstance(d, dict)

    def test_post_parameter_update(self):
        """Test whether post_parameter_update resets the regularization term."""
        # set regularizer term
        self.model.regularizer.regularization_term = None

        # call post_parameter_update
        self.model.post_parameter_update()

        # assert that the regularization term has been reset
        assert self.model.regularizer.regularization_term == torch.zeros(1, dtype=torch.float)


class _DistanceModelTestCase(_ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class TestComplex(_ModelTestCase, unittest.TestCase):
    """Test the ComplEx model."""

    model_cls = poem.models.ComplEx


class TestConvE(_ModelTestCase, unittest.TestCase):
    """Test the ConvE model."""

    model_cls = poem.models.ConvE
    embedding_dim = 12
    create_inverse_triples = True
    model_kwargs = {
        'output_channels': 2,
        'embedding_height': 3,
        'embedding_width': 4,
    }


class TestConvKB(_ModelTestCase, unittest.TestCase):
    """Test the ConvKB model."""

    model_cls = poem.models.ConvKB
    model_kwargs = {
        'num_filters': 2,
    }


class TestDistMult(_ModelTestCase, unittest.TestCase):
    """Test the DistMult model."""

    model_cls = poem.models.DistMult


class TestERMLP(_ModelTestCase, unittest.TestCase):
    """Test the ERMLP model."""

    model_cls = poem.models.ERMLP
    model_kwargs = {
        'hidden_dim': 4,
    }


class TestERMLPE(_ModelTestCase, unittest.TestCase):
    """Test the extended ERMLP model."""

    model_cls = poem.models.ERMLPE
    model_kwargs = {
        'hidden_dim': 4,
    }


class TestHolE(_ModelTestCase, unittest.TestCase):
    """Test the HolE model."""

    model_cls = poem.models.HolE


class TestCaseKG2EWithKL(_ModelTestCase, unittest.TestCase):
    """Test the KG2E model with KL similarity."""

    model_cls = poem.models.KG2E
    model_kwargs = {
        'dist_similarity': 'KL',
    }


class TestCaseKG2EWithEL(_ModelTestCase, unittest.TestCase):
    """Test the KG2E model with EL similarity."""

    model_cls = poem.models.KG2E
    model_kwargs = {
        'dist_similarity': 'EL',
    }


class TestNTN(_ModelTestCase, unittest.TestCase):
    """Test the NTN model."""

    model_cls = poem.models.NTN
    model_kwargs = {
        'num_slices': 2,
    }


class TestProjE(_ModelTestCase, unittest.TestCase):
    """Test the ProjE model."""

    model_cls = poem.models.ProjE


class TestRESCAL(_ModelTestCase, unittest.TestCase):
    """Test the RESCAL model."""

    model_cls = poem.models.RESCAL


class TestRGCN(_ModelTestCase, unittest.TestCase):
    """Test the R-GCN model."""

    model_cls = poem.models.RGCN
    sampler = 'schlichtkrull'


class TestRGCNBlock(_ModelTestCase, unittest.TestCase):
    """Test the R-GCN model with block decomposition."""

    model_cls = poem.models.RGCN
    sampler = 'schlichtkrull'
    embedding_dim = 6
    model_kwargs = {
        'decomposition': 'block',
        'num_bases_or_blocks': 3,
        'message_normalization': 'symmetric',
    }


class TestRotatE(_ModelTestCase, unittest.TestCase):
    """Test the RotatE model."""

    model_cls = poem.models.RotatE


class TestSimplE(_ModelTestCase, unittest.TestCase):
    """Test the SimplE model."""

    model_cls = poem.models.SimplE


class TestSE(_ModelTestCase, unittest.TestCase):
    """Test the Structured Embedding model."""

    model_cls = poem.models.StructuredEmbedding


class TestTransD(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransD model."""

    model_cls = poem.models.TransD
    model_kwargs = {
        'relation_dim': 4,
    }


class TestTransE(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransE model."""

    model_cls = poem.models.TransE


class TestTransH(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransH model."""

    model_cls = poem.models.TransH


class TestTransR(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransR model."""

    model_cls = poem.models.TransR
    model_kwargs = {
        'relation_dim': 4,
    }


class TestTuckEr(_ModelTestCase, unittest.TestCase):
    """Test the TuckEr model."""

    model_cls = poem.models.TuckER
    model_kwargs = {
        'relation_dim': 4,
    }


class TestUM(_DistanceModelTestCase, unittest.TestCase):
    """Test the Unstructured Model."""

    model_cls = poem.models.UnstructuredModel


class TestTesting(unittest.TestCase):
    """Yo dawg, I heard you like testing, so I wrote a test to test the tests so you can test while you're testing."""

    def test_testing(self):
        """Check that there's a test for all models.

        For now, this is excluding multimodel models. Not sure how to test those yet.
        """
        model_names = {
            cls.__name__
            for cls in BaseModule.__subclasses__()
        }
        model_names -= SKIP_MODULES

        tested_model_names = {
            value.model_cls.__name__
            for name, value in globals().items()
            if (
                isinstance(value, type)
                and issubclass(value, _ModelTestCase)
                and not name.startswith('_')
                and not issubclass(value.model_cls, MultimodalBaseModule)
            )
        }
        tested_model_names -= SKIP_MODULES

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
            self.assertIn(f':class:`poem.models.{name}`', poem.models.__doc__, msg=f'Forgot to document {name}')

    def test_models_have_experiments(self):
        """Test that each model has an experiment folder in :mod:`poem.experiments`."""
        experiments_path = os.path.abspath(os.path.dirname(poem.experiments.__file__))
        experiment_blacklist = {
            'DistMultLiteral',  # FIXME
            'ComplExLiteral',  # FIXME
            'UnstructuredModel',
            'StructuredEmbedding',
            'RESCAL',
            'NTN',
            'ERMLP',
            'ConvE',  # FIXME
            'ProjE',  # FIXME
            'ERMLPE',  # FIXME
        }
        model_names = set(poem.models.__all__) - SKIP_MODULES - experiment_blacklist
        missing = {
            model
            for model in model_names
            if not os.path.exists(os.path.join(experiments_path, model.lower()))
        }
        if missing:
            _s = '\n'.join(f'- [ ] {model.lower()}' for model in sorted(missing))
            self.fail(f'Missing experimental configuration directories for the following models:\n{_s}')
