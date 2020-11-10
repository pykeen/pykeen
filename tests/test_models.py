# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import tempfile
import traceback
import unittest
from typing import Any, ClassVar, Mapping, Optional, Type
from unittest.mock import patch

import numpy
import pytest
import torch
from click.testing import CliRunner, Result
from torch import nn, optim
from torch.optim import SGD
from torch.optim.adagrad import Adagrad

import pykeen.experiments
import pykeen.models
from pykeen.datasets.kinships import KINSHIPS_TRAIN_PATH
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH, Nations
from pykeen.models import _MODELS
from pykeen.models.base import (
    EntityEmbeddingModel,
    EntityRelationEmbeddingModel,
    Model,
    MultimodalModel,
    _extend_batch,
    get_novelty_mask,
)
from pykeen.models.cli import build_cli_from_cls
from pykeen.models.unimodal.rgcn import (
    inverse_indegree_edge_weights,
    inverse_outdegree_edge_weights,
    symmetric_edge_weights,
)
from pykeen.models.unimodal.trans_d import _project_entity
from pykeen.nn import Embedding, RepresentationModule
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop, TrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.utils import all_in_bounds, clamp_norm, set_random_seed

SKIP_MODULES = {
    Model.__name__,
    'DummyModel',
    MultimodalModel.__name__,
    EntityEmbeddingModel.__name__,
    EntityRelationEmbeddingModel.__name__,
    'MockModel',
    'models',
    'get_model_cls',
    'SimpleInteractionModel',
}
for cls in MultimodalModel.__subclasses__():
    SKIP_MODULES.add(cls.__name__)

_EPSILON = 1.0e-07


class _CustomRepresentations(RepresentationModule):
    """A custom representation module with minimal implementation."""

    def __init__(self, num_entities: int, embedding_dim: int = 2):
        super().__init__()
        self.num_embeddings = num_entities
        self.embedding_dim = embedding_dim
        self.x = nn.Parameter(torch.rand(embedding_dim))

    def forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        n = self.num_embeddings if indices is None else indices.shape[0]
        return self.x.unsqueeze(dim=0).repeat(n, 1)

    def get_in_canonical_shape(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        x = self(indices=indices)
        if indices is None:
            return x.unsqueeze(dim=0)
        return x.unsqueeze(dim=1)


class _ModelTestCase:
    """A test case for quickly defining common tests for KGE models."""

    #: The class of the model to test
    model_cls: ClassVar[Type[Model]]

    #: Additional arguments passed to the model's constructor method
    model_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    #: The triples factory instance
    factory: TriplesFactory

    #: The model instance
    model: Model

    #: The batch size for use for forward_* tests
    batch_size: int = 20

    #: The embedding dimensionality
    embedding_dim: int = 3

    #: Whether to create inverse triples (needed e.g. by ConvE)
    create_inverse_triples: bool = False

    #: The sampler to use for sLCWA (different e.g. for R-GCN)
    sampler = 'default'

    #: The batch size for use when testing training procedures
    train_batch_size = 400

    #: The number of epochs to train the model
    train_num_epochs = 2

    #: A random number generator from torch
    generator: torch.Generator

    #: The number of parameters which receive a constant (i.e. non-randomized)
    # initialization
    num_constant_init: int = 0

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        _, self.generator, _ = set_random_seed(42)

        dataset = Nations(create_inverse_triples=self.create_inverse_triples)
        self.factory = dataset.training
        self.model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            **(self.model_kwargs or {}),
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

    def test_reset_parameters_(self):
        """Test :func:`Model.reset_parameters_`."""
        # get model parameters
        params = list(self.model.parameters())
        old_content = {
            id(p): p.data.detach().clone()
            for p in params
        }

        # re-initialize
        self.model.reset_parameters_()

        # check that the operation works in-place
        new_params = list(self.model.parameters())
        assert set(id(np) for np in new_params) == set(id(p) for p in params)

        # check that the parameters where modified
        num_equal_weights_after_re_init = sum(
            1
            for np in new_params
            if (np.data == old_content[id(np)]).all()
        )
        assert num_equal_weights_after_re_init == self.num_constant_init, (
            num_equal_weights_after_re_init, self.num_constant_init,
        )

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        self.assertTrue(torch.all(torch.isfinite(scores)).item(), f'Some scores were not finite:\n{scores}')

        # check whether a gradient can be back-propgated
        scores.mean().backward()

    def test_score_hrt(self) -> None:
        """Test the model's ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.model.device)
        try:
            scores = self.model.score_hrt(batch)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(batch, scores)

    def test_score_t(self) -> None:
        """Test the model's ``score_t()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :2].to(self.model.device)
        # assert batch comprises (head, relation) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores = self.model.score_t(batch)
        except NotImplementedError:
            self.fail(msg='Score_o not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    def test_score_h(self) -> None:
        """Test the model's ``score_h()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.model.device)
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores = self.model.score_h(batch)
        except NotImplementedError:
            self.fail(msg='Score_s not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    @pytest.mark.slow
    def test_train_slcwa(self) -> None:
        """Test that sLCWA training does not fail."""
        loop = SLCWATrainingLoop(
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

    @pytest.mark.slow
    def test_train_lcwa(self) -> None:
        """Test that LCWA training does not fail."""
        loop = LCWATrainingLoop(
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

    def test_save_load_model_state(self):
        """Test whether a saved model state can be re-loaded."""
        original_model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            random_seed=42,
            **(self.model_kwargs or {}),
        ).to_device_()

        loaded_model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            random_seed=21,
            **(self.model_kwargs or {}),
        ).to_device_()

        def _equal_embeddings(a: RepresentationModule, b: RepresentationModule) -> bool:
            """Test whether two embeddings are equal."""
            return (a(indices=None) == b(indices=None)).all()

        if isinstance(original_model, EntityEmbeddingModel):
            assert not _equal_embeddings(original_model.entity_embeddings, loaded_model.entity_embeddings)
        if isinstance(original_model, EntityRelationEmbeddingModel):
            assert not _equal_embeddings(original_model.relation_embeddings, loaded_model.relation_embeddings)

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, 'test.pt')
            original_model.save_state(path=file_path)
            loaded_model.load_state(path=file_path)
        if isinstance(original_model, EntityEmbeddingModel):
            assert _equal_embeddings(original_model.entity_embeddings, loaded_model.entity_embeddings)
        if isinstance(original_model, EntityRelationEmbeddingModel):
            assert _equal_embeddings(original_model.relation_embeddings, loaded_model.relation_embeddings)

    @property
    def cli_extras(self):
        kwargs = self.model_kwargs or {}
        extras = [
            '--silent',
        ]
        for k, v in kwargs.items():
            extras.append('--' + k.replace('_', '-'))
            extras.append(str(v))
        extras += [
            '--number-epochs', self.train_num_epochs,
            '--embedding-dim', self.embedding_dim,
            '--batch-size', self.train_batch_size,
        ]
        extras = [str(e) for e in extras]
        return extras

    @pytest.mark.slow
    def test_cli_training_nations(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH] + self.cli_extras)

    @pytest.mark.slow
    def test_cli_training_kinships(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', KINSHIPS_TRAIN_PATH] + self.cli_extras)

    @pytest.mark.slow
    def test_cli_training_nations_testing(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH, '-q', NATIONS_TEST_PATH] + self.cli_extras)

    def _help_test_cli(self, args):
        """Test running the pipeline on all models."""
        if issubclass(self.model_cls, pykeen.models.RGCN):
            self.skipTest('There is a problem with non-reproducible unittest for R-GCN.')
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
$ pykeen train {self.model_cls.__name__.lower()} {' '.join(args)}

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

    def test_post_parameter_update_regularizer(self):
        """Test whether post_parameter_update resets the regularization term."""
        # set regularizer term
        self.model.regularizer.regularization_term = None

        # call post_parameter_update
        self.model.post_parameter_update()

        # assert that the regularization term has been reset
        assert self.model.regularizer.regularization_term == torch.zeros(1, dtype=torch.float, device=self.model.device)

    def test_post_parameter_update(self):
        """Test whether post_parameter_update correctly enforces model constraints."""
        # do one optimization step
        opt = optim.SGD(params=self.model.parameters(), lr=1.)
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.model.device)
        scores = self.model.score_hrt(hrt_batch=batch)
        fake_loss = scores.mean()
        fake_loss.backward()
        opt.step()

        # call post_parameter_update
        self.model.post_parameter_update()

        # check model constraints
        self._check_constraints()

    def _check_constraints(self):
        """Check model constraints."""

    def test_score_h_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_h()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.model.device)
        self.model.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_h = self.model.score_h(batch)
            scores_hrt = super(self.model.__class__, self.model).score_h(batch)
        except NotImplementedError:
            self.fail(msg='Score_h not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        assert torch.allclose(scores_h, scores_hrt, atol=1e-06)

    def test_score_r_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_r()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, [0, 2]].to(self.model.device)
        self.model.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_r = self.model.score_r(batch)
            scores_hrt = super(self.model.__class__, self.model).score_r(batch)
        except NotImplementedError:
            self.fail(msg='Score_h not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        assert torch.allclose(scores_r, scores_hrt, atol=1e-06)

    def test_score_t_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_t()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :-1].to(self.model.device)
        self.model.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores_t = self.model.score_t(batch)
            scores_hrt = super(self.model.__class__, self.model).score_t(batch)
        except NotImplementedError:
            self.fail(msg='Score_h not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        assert torch.allclose(scores_t, scores_hrt, atol=1e-06)

    def test_reset_parameters_constructor_call(self):
        """Tests whether reset_parameters is called in the constructor."""
        with patch.object(self.model_cls, 'reset_parameters_', return_value=None) as mock_method:
            try:
                self.model_cls(
                    triples_factory=self.factory,
                    embedding_dim=self.embedding_dim,
                    **(self.model_kwargs or {}),
                )
            except TypeError as error:
                assert error.args == ("'NoneType' object is not callable",)
            mock_method.assert_called_once()

    def test_custom_representations(self):
        """Tests whether we can provide custom representations."""
        if isinstance(self.model, EntityEmbeddingModel):
            old_embeddings = self.model.entity_embeddings
            self.model.entity_embeddings = _CustomRepresentations(
                num_entities=self.factory.num_entities,
                embedding_dim=old_embeddings.embedding_dim,
            )
            # call some functions
            self.model.reset_parameters_()
            self.test_score_hrt()
            self.test_score_t()
            # reset to old state
            self.model.entity_embeddings = old_embeddings
        elif isinstance(self.model, EntityRelationEmbeddingModel):
            old_embeddings = self.model.relation_embeddings
            self.model.relation_embeddings = _CustomRepresentations(
                num_entities=self.factory.num_relations,
                embedding_dim=old_embeddings.embedding_dim,
            )
            # call some functions
            self.model.reset_parameters_()
            self.test_score_hrt()
            self.test_score_t()
            # reset to old state
            self.model.relation_embeddings = old_embeddings
        else:
            self.skipTest(f'Not testing custom representations for model: {self.model.__class__.__name__}')


class _DistanceModelTestCase(_ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class TestComplex(_ModelTestCase, unittest.TestCase):
    """Test the ComplEx model."""

    model_cls = pykeen.models.ComplEx


class TestConvE(_ModelTestCase, unittest.TestCase):
    """Test the ConvE model."""

    model_cls = pykeen.models.ConvE
    embedding_dim = 12
    create_inverse_triples = True
    model_kwargs = {
        'output_channels': 2,
        'embedding_height': 3,
        'embedding_width': 4,
    }
    # 3x batch norm: bias + scale --> 6
    # entity specific bias        --> 1
    # ==================================
    #                                 7
    num_constant_init = 7


class TestConvKB(_ModelTestCase, unittest.TestCase):
    """Test the ConvKB model."""

    model_cls = pykeen.models.ConvKB
    model_kwargs = {
        'num_filters': 2,
    }
    # two bias terms, one conv-filter
    num_constant_init = 3


class TestDistMult(_ModelTestCase, unittest.TestCase):
    """Test the DistMult model."""

    model_cls = pykeen.models.DistMult

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.model.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))

    def _test_score_all_triples(self, k: Optional[int], batch_size: int = 16):
        """Test score_all_triples.

        :param k: The number of triples to return. Set to None, to keep all.
        :param batch_size: The batch size to use for calculating scores.
        """
        top_triples, top_scores = self.model.score_all_triples(k=k, batch_size=batch_size, return_tensors=True)

        # check type
        assert torch.is_tensor(top_triples)
        assert torch.is_tensor(top_scores)
        assert top_triples.dtype == torch.long
        assert top_scores.dtype == torch.float32

        # check shape
        actual_k, n_cols = top_triples.shape
        assert n_cols == 3
        if k is None:
            assert actual_k == self.factory.num_entities ** 2 * self.factory.num_relations
        else:
            assert actual_k == min(k, self.factory.num_triples)
        assert top_scores.shape == (actual_k,)

        # check ID ranges
        assert (top_triples >= 0).all()
        assert top_triples[:, [0, 2]].max() < self.model.num_entities
        assert top_triples[:, 1].max() < self.model.num_relations

    def test_score_all_triples(self):
        """Test score_all_triples with a large batch size."""
        # this is only done in one of the models
        self._test_score_all_triples(k=15, batch_size=16)

    def test_score_all_triples_singleton_batch(self):
        """Test score_all_triples with a batch size of 1."""
        self._test_score_all_triples(k=15, batch_size=1)

    def test_score_all_triples_large_batch(self):
        """Test score_all_triples with a batch size larger than k."""
        self._test_score_all_triples(k=10, batch_size=16)

    def test_score_all_triples_keep_all(self):
        """Test score_all_triples with k=None."""
        # this is only done in one of the models
        self._test_score_all_triples(k=None)


class TestERMLP(_ModelTestCase, unittest.TestCase):
    """Test the ERMLP model."""

    model_cls = pykeen.models.ERMLP
    model_kwargs = {
        'hidden_dim': 4,
    }
    # Two linear layer biases
    num_constant_init = 2


class TestERMLPE(_ModelTestCase, unittest.TestCase):
    """Test the extended ERMLP model."""

    model_cls = pykeen.models.ERMLPE
    model_kwargs = {
        'hidden_dim': 4,
    }
    # Two BN layers, bias & scale
    num_constant_init = 4


class TestHolE(_ModelTestCase, unittest.TestCase):
    """Test the HolE model."""

    model_cls = pykeen.models.HolE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have at most unit L2 norm.
        """
        assert all_in_bounds(self.model.entity_embeddings(indices=None).norm(p=2, dim=-1), high=1., a_tol=_EPSILON)


class _TestKG2E(_ModelTestCase):
    """General tests for the KG2E model."""

    model_cls = pykeen.models.KG2E

    def _check_constraints(self):
        """Check model constraints.

        * Entity and relation embeddings have to have at most unit L2 norm.
        * Covariances have to have values between c_min and c_max
        """
        for embedding in (self.model.entity_embeddings, self.model.relation_embeddings):
            assert all_in_bounds(embedding(indices=None).norm(p=2, dim=-1), high=1., a_tol=_EPSILON)
        for cov in (self.model.entity_covariances, self.model.relation_covariances):
            assert all_in_bounds(cov(indices=None), low=self.model.c_min, high=self.model.c_max)


class TestKG2EWithKL(_TestKG2E, unittest.TestCase):
    """Test the KG2E model with KL similarity."""

    model_kwargs = {
        'dist_similarity': 'KL',
    }


class TestKG2EWithEL(_TestKG2E, unittest.TestCase):
    """Test the KG2E model with EL similarity."""

    model_kwargs = {
        'dist_similarity': 'EL',
    }


class _BaseNTNTest(_ModelTestCase, unittest.TestCase):
    """Test the NTN model."""

    model_cls = pykeen.models.NTN

    def test_can_slice(self):
        """Test that the slicing properties are calculated correctly."""
        self.assertTrue(self.model.can_slice_h)
        self.assertFalse(self.model.can_slice_r)
        self.assertTrue(self.model.can_slice_t)


class TestNTNLowMemory(_BaseNTNTest):
    """Test the NTN model with automatic memory optimization."""

    model_kwargs = {
        'num_slices': 2,
        'automatic_memory_optimization': True,
    }


class TestNTNHighMemory(_BaseNTNTest):
    """Test the NTN model without automatic memory optimization."""

    model_kwargs = {
        'num_slices': 2,
        'automatic_memory_optimization': False,
    }


class TestProjE(_ModelTestCase, unittest.TestCase):
    """Test the ProjE model."""

    model_cls = pykeen.models.ProjE


class TestRESCAL(_ModelTestCase, unittest.TestCase):
    """Test the RESCAL model."""

    model_cls = pykeen.models.RESCAL


class _TestRGCN(_ModelTestCase):
    """Test the R-GCN model."""

    model_cls = pykeen.models.RGCN
    sampler = 'schlichtkrull'

    def _check_constraints(self):
        """Check model constraints.

        Enriched embeddings have to be reset.
        """
        assert self.model.entity_representations.enriched_embeddings is None


class TestRGCNBasis(_TestRGCN, unittest.TestCase):
    """Test the R-GCN model."""

    model_kwargs = {
        'decomposition': 'basis',
    }
    #: one bias per layer
    num_constant_init = 2


class TestRGCNBlock(_TestRGCN, unittest.TestCase):
    """Test the R-GCN model with block decomposition."""

    embedding_dim = 6
    model_kwargs = {
        'decomposition': 'block',
        'num_bases_or_blocks': 3,
        'edge_weighting': symmetric_edge_weights,
        'use_batch_norm': True,
    }
    #: (scale & bias for BN) * layers
    num_constant_init = 4


class TestRotatE(_ModelTestCase, unittest.TestCase):
    """Test the RotatE model."""

    model_cls = pykeen.models.RotatE

    def _check_constraints(self):
        """Check model constraints.

        Relation embeddings' entries have to have absolute value 1 (i.e. represent a rotation in complex plane)
        """
        relation_abs = (
            self.model
                .relation_embeddings(indices=None)
                .view(self.factory.num_relations, -1, 2)
                .norm(p=2, dim=-1)
        )
        assert torch.allclose(relation_abs, torch.ones_like(relation_abs))


class TestSimplE(_ModelTestCase, unittest.TestCase):
    """Test the SimplE model."""

    model_cls = pykeen.models.SimplE


class _BaseTestSE(_ModelTestCase, unittest.TestCase):
    """Test the Structured Embedding model."""

    model_cls = pykeen.models.StructuredEmbedding

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        norms = self.model.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))


class TestSELowMemory(_BaseTestSE):
    """Tests SE with low memory."""

    model_kwargs = dict(
        automatic_memory_optimization=True,
    )


class TestSEHighMemory(_BaseTestSE):
    """Tests SE with low memory."""

    model_kwargs = dict(
        automatic_memory_optimization=False,
    )


class TestTransD(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransD model."""

    model_cls = pykeen.models.TransD
    model_kwargs = {
        'relation_dim': 4,
    }

    def _check_constraints(self):
        """Check model constraints.

        Entity and relation embeddings have to have at most unit L2 norm.
        """
        for emb in (self.model.entity_embeddings, self.model.relation_embeddings):
            assert all_in_bounds(emb(indices=None).norm(p=2, dim=-1), high=1., a_tol=_EPSILON)

    def test_score_hrt_manual(self):
        """Manually test interaction function of TransD."""
        # entity embeddings
        weights = torch.as_tensor(data=[[2., 2.], [4., 4.]], dtype=torch.float)
        entity_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_embeddings._embeddings.weight.data.copy_(weights)
        self.model.entity_embeddings = entity_embeddings

        projection_weights = torch.as_tensor(data=[[3., 3.], [2., 2.]], dtype=torch.float)
        entity_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_projection_embeddings._embeddings.weight.data.copy_(projection_weights)
        self.model.entity_projections = entity_projection_embeddings

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[4.], [4.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=1,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.model.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[5.], [3.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=1,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.model.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        scores = self.model.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        self.assertAlmostEqual(first_score, -16, delta=0.01)

        # Use different dimension for relation embedding: relation_dim > entity_dim
        # relation embeddings
        relation_weights = torch.as_tensor(data=[[3., 3., 3.], [3., 3., 3.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.model.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[4., 4., 4.], [4., 4., 4.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.model.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0]], dtype=torch.long)
        scores = self.model.score_hrt(hrt_batch=batch)
        self.assertAlmostEqual(scores.item(), -27, delta=0.01)

        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        scores = self.model.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        second_score = scores[1].item()
        self.assertAlmostEqual(first_score, -27, delta=0.01)
        self.assertAlmostEqual(second_score, -27, delta=0.01)

        # Use different dimension for relation embedding: relation_dim < entity_dim
        # entity embeddings
        weights = torch.as_tensor(data=[[1., 1., 1.], [1., 1., 1.]], dtype=torch.float)
        entity_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        entity_embeddings._embeddings.weight.data.copy_(weights)
        self.model.entity_embeddings = entity_embeddings

        projection_weights = torch.as_tensor(data=[[2., 2., 2.], [2., 2., 2.]], dtype=torch.float)
        entity_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        entity_projection_embeddings._embeddings.weight.data.copy_(projection_weights)
        self.model.entity_projections = entity_projection_embeddings

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[3., 3.], [3., 3.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.model.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[4., 4.], [4., 4.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.model.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        scores = self.model.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        second_score = scores[1].item()
        self.assertAlmostEqual(first_score, -18, delta=0.01)
        self.assertAlmostEqual(second_score, -18, delta=0.01)

    def test_project_entity(self):
        """Test _project_entity."""
        # random entity embeddings & projections
        e = torch.rand(1, self.model.num_entities, self.embedding_dim, generator=self.generator)
        e = clamp_norm(e, maxnorm=1, p=2, dim=-1)
        e_p = torch.rand(1, self.model.num_entities, self.embedding_dim, generator=self.generator)

        # random relation embeddings & projections
        r = torch.rand(self.batch_size, 1, self.model.relation_dim, generator=self.generator)
        r = clamp_norm(r, maxnorm=1, p=2, dim=-1)
        r_p = torch.rand(self.batch_size, 1, self.model.relation_dim, generator=self.generator)

        # project
        e_bot = _project_entity(e=e, e_p=e_p, r=r, r_p=r_p)

        # check shape:
        assert e_bot.shape == (self.batch_size, self.model.num_entities, self.model.relation_dim)

        # check normalization
        assert (torch.norm(e_bot, dim=-1, p=2) <= 1.0 + 1.0e-06).all()


class TestTransE(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransE model."""

    model_cls = pykeen.models.TransE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.model.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransH(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransH model."""

    model_cls = pykeen.models.TransH

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.model.normal_vector_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransR(_DistanceModelTestCase, unittest.TestCase):
    """Test the TransR model."""

    model_cls = pykeen.models.TransR
    model_kwargs = {
        'relation_dim': 4,
    }

    def test_score_hrt_manual(self):
        """Manually test interaction function of TransR."""
        # entity embeddings
        weights = torch.as_tensor(data=[[2., 2.], [3., 3.]], dtype=torch.float)
        entity_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_embeddings._embeddings.weight.data.copy_(weights)
        self.model.entity_embeddings = entity_embeddings

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[4., 4], [5., 5.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.model.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[5., 5., 6., 6.], [7., 7., 8., 8.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=4,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.model.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        scores = self.model.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        # second_score = scores[1].item()
        self.assertAlmostEqual(first_score, -32, delta=0.01)

    def _check_constraints(self):
        """Check model constraints.

        Entity and relation embeddings have to have at most unit L2 norm.
        """
        for emb in (self.model.entity_embeddings, self.model.relation_embeddings):
            assert all_in_bounds(emb(indices=None).norm(p=2, dim=-1), high=1., a_tol=1.0e-06)


class TestTuckEr(_ModelTestCase, unittest.TestCase):
    """Test the TuckEr model."""

    model_cls = pykeen.models.TuckER
    model_kwargs = {
        'relation_dim': 4,
    }
    #: 2xBN (bias & scale)
    num_constant_init = 4


class TestUM(_DistanceModelTestCase, unittest.TestCase):
    """Test the Unstructured Model."""

    model_cls = pykeen.models.UnstructuredModel


class TestTesting(unittest.TestCase):
    """Yo dawg, I heard you like testing, so I wrote a test to test the tests so you can test while you're testing."""

    def test_testing(self):
        """Check that there's a test for all models.

        For now, this is excluding multimodel models. Not sure how to test those yet.
        """
        model_names = {
            cls.__name__
            for cls in pykeen.models.models.values()
        }
        model_names -= SKIP_MODULES

        tested_model_names = {
            value.model_cls.__name__
            for name, value in globals().items()
            if (
                isinstance(value, type)
                and issubclass(value, _ModelTestCase)
                and not name.startswith('_')
                and not issubclass(value.model_cls, MultimodalModel)
            )
        }
        tested_model_names -= SKIP_MODULES

        self.assertEqual(model_names, tested_model_names, msg='Some models have not been tested')

    def test_importing(self):
        """Test that all models are available from :mod:`pykeen.models`."""
        models_path = os.path.abspath(os.path.dirname(pykeen.models.__file__))

        model_names = set()
        for directory, _, filenames in os.walk(models_path):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue

                path = os.path.join(directory, filename)
                relpath = os.path.relpath(path, models_path)
                if relpath.endswith('__init__.py'):
                    continue

                import_path = 'pykeen.models.' + relpath[:-len('.py')].replace(os.sep, '.')
                module = importlib.import_module(import_path)

                for name in dir(module):
                    value = getattr(module, name)
                    if (
                        isinstance(value, type)
                        and issubclass(value, Model)
                    ):
                        model_names.add(value.__name__)

        star_model_names = set(pykeen.models.__all__) - SKIP_MODULES
        model_names -= SKIP_MODULES

        self.assertEqual(model_names, star_model_names, msg='Forgot to add some imports')

    def test_models_have_experiments(self):
        """Test that each model has an experiment folder in :mod:`pykeen.experiments`."""
        experiments_path = os.path.abspath(os.path.dirname(pykeen.experiments.__file__))
        experiment_blacklist = {
            'DistMultLiteral',  # FIXME
            'ComplExLiteral',  # FIXME
            'UnstructuredModel',
            'StructuredEmbedding',
            'RESCAL',
            'NTN',
            'ERMLP',
            'ProjE',  # FIXME
            'ERMLPE',  # FIXME
        }
        model_names = set(pykeen.models.__all__) - SKIP_MODULES - experiment_blacklist
        missing = {
            model
            for model in model_names
            if not os.path.exists(os.path.join(experiments_path, model.lower()))
        }
        if missing:
            _s = '\n'.join(f'- [ ] {model.lower()}' for model in sorted(missing))
            self.fail(f'Missing experimental configuration directories for the following models:\n{_s}')


def test_extend_batch():
    """Test `_extend_batch()`."""
    batch = torch.tensor([[a, b] for a in range(3) for b in range(4)]).view(-1, 2)
    all_ids = [2 * i for i in range(5)]

    batch_size = batch.shape[0]
    num_choices = len(all_ids)

    for dim in range(3):
        h_ext_batch = _extend_batch(batch=batch, all_ids=all_ids, dim=dim)

        # check shape
        assert h_ext_batch.shape == (batch_size * num_choices, 3)

        # check content
        actual_content = set(tuple(map(int, hrt)) for hrt in h_ext_batch)
        exp_content = set()
        for i in all_ids:
            for b in batch:
                c = list(map(int, b))
                c.insert(dim, i)
                exp_content.add(tuple(c))

        assert actual_content == exp_content


class MessageWeightingTests(unittest.TestCase):
    """unittests for message weighting."""

    #: The number of entities
    num_entities: int = 16

    #: The number of triples
    num_triples: int = 101

    def setUp(self) -> None:
        """Initialize data for unittest."""
        self.source, self.target = torch.randint(self.num_entities, size=(2, self.num_triples))

    def _test_message_weighting(self, weight_func):
        """Perform common tests for message weighting."""
        weights = weight_func(source=self.source, target=self.target)

        # check shape
        assert weights.shape == self.source.shape

        # check dtype
        assert weights.dtype == torch.float32

        # check finite values (e.g. due to division by zero)
        assert torch.isfinite(weights).all()

        # check non-negativity
        assert (weights >= 0.).all()

    def test_inverse_indegree_edge_weights(self):
        """Test inverse_indegree_edge_weights."""
        self._test_message_weighting(weight_func=inverse_indegree_edge_weights)

    def test_inverse_outdegree_edge_weights(self):
        """Test inverse_outdegree_edge_weights."""
        self._test_message_weighting(weight_func=inverse_outdegree_edge_weights)

    def test_symmetric_edge_weights(self):
        """Test symmetric_edge_weights."""
        self._test_message_weighting(weight_func=symmetric_edge_weights)


def test_get_novelty_mask():
    """Test `get_novelty_mask()`."""
    num_triples = 7
    base = torch.arange(num_triples)
    mapped_triples = torch.stack([base, base, 3 * base], dim=-1)
    query_ids = torch.randperm(num_triples).numpy()[:num_triples // 2]
    exp_novel = query_ids != 0
    col = 2
    other_col_ids = numpy.asarray([0, 0])
    mask = get_novelty_mask(
        mapped_triples=mapped_triples,
        query_ids=query_ids,
        col=col,
        other_col_ids=other_col_ids,
    )
    assert mask.shape == query_ids.shape
    assert (mask == exp_novel).all()


class TestRandom(unittest.TestCase):
    """Extra tests."""

    def test_abstract(self):
        """Test that classes are checked as abstract properly."""
        self.assertTrue(Model._is_abstract())
        self.assertTrue(EntityEmbeddingModel._is_abstract())
        self.assertTrue(EntityRelationEmbeddingModel._is_abstract())
        for model_cls in _MODELS:
            if issubclass(model_cls, MultimodalModel):
                continue
            self.assertFalse(model_cls._is_abstract(), msg=f'{model_cls.__name__} should not be abstract')
