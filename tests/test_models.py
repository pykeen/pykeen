# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import unittest
from typing import Any, MutableMapping, Optional

import numpy
import torch
import unittest_templates

import pykeen.experiments
import pykeen.models
from pykeen.models import (
    ERModel, EntityEmbeddingModel, EntityRelationEmbeddingModel, Model,
    _NewAbstractModel, _OldAbstractModel, model_resolver,
)
from pykeen.models.multimodal.base import LiteralModel
from pykeen.models.predict import get_novelty_mask, predict
from pykeen.models.unimodal.trans_d import _project_entity
from pykeen.nn import EmbeddingSpecification
from pykeen.nn.emb import Embedding
from pykeen.utils import all_in_bounds, clamp_norm, extend_batch
from tests import cases
from tests.constants import EPSILON
from tests.mocks import MockModel
from tests.test_model_mode import SimpleInteractionModel

SKIP_MODULES = {
    Model,
    _OldAbstractModel,
    _NewAbstractModel,
    # DummyModel,
    LiteralModel,
    EntityEmbeddingModel,
    EntityRelationEmbeddingModel,
    ERModel,
    MockModel,
    SimpleInteractionModel,
}
SKIP_MODULES.update(LiteralModel.__subclasses__())


class TestCompGCN(cases.ModelTestCase):
    """Test the CompGCN model."""

    cls = pykeen.models.CompGCN
    create_inverse_triples = True
    num_constant_init = 3  # BN(2) + Bias
    cli_extras = ['--create-inverse-triples']

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["encoder_kwargs"] = dict(
            embedding_specification=EmbeddingSpecification(
                embedding_dim=(kwargs.pop("embedding_dim")),
            ),
        )
        return kwargs


class TestComplex(cases.ModelTestCase):
    """Test the ComplEx model."""

    cls = pykeen.models.ComplEx


class TestConvE(cases.ModelTestCase):
    """Test the ConvE model."""

    cls = pykeen.models.ConvE
    embedding_dim = 12
    create_inverse_triples = True
    kwargs = {
        'output_channels': 2,
        'embedding_height': 3,
        'embedding_width': 4,
    }
    # 3x batch norm: bias + scale --> 6
    # entity specific bias        --> 1
    # ==================================
    #                                 7
    num_constant_init = 7


class TestConvKB(cases.ModelTestCase):
    """Test the ConvKB model."""

    cls = pykeen.models.ConvKB
    kwargs = {
        'num_filters': 2,
    }
    # two bias terms, one conv-filter
    num_constant_init = 3


class TestDistMult(cases.ModelTestCase):
    """Test the DistMult model."""

    cls = pykeen.models.DistMult

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.instance.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))

    def _test_score_all_triples(self, k: Optional[int], batch_size: int = 16):
        """Test score_all_triples.

        :param k: The number of triples to return. Set to None, to keep all.
        :param batch_size: The batch size to use for calculating scores.
        """
        top_triples, top_scores = predict(model=self.instance, batch_size=batch_size, k=k)

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
        assert top_triples[:, [0, 2]].max() < self.instance.num_entities
        assert top_triples[:, 1].max() < self.instance.num_relations

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


class TestERMLP(cases.ModelTestCase):
    """Test the ERMLP model."""

    cls = pykeen.models.ERMLP
    kwargs = {
        'hidden_dim': 4,
    }
    # Two linear layer biases
    num_constant_init = 2


class TestERMLPE(cases.ModelTestCase):
    """Test the extended ERMLP model."""

    cls = pykeen.models.ERMLPE
    kwargs = {
        'hidden_dim': 4,
    }
    # Two BN layers, bias & scale
    num_constant_init = 4


class TestHolE(cases.ModelTestCase):
    """Test the HolE model."""

    cls = pykeen.models.HolE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have at most unit L2 norm.
        """
        assert all_in_bounds(self.instance.entity_embeddings(indices=None).norm(p=2, dim=-1), high=1., a_tol=EPSILON)


class TestKG2EWithKL(cases.BaseKG2ETest):
    """Test the KG2E model with KL similarity."""

    kwargs = {
        'dist_similarity': 'KL',
    }


class TestMuRE(cases.ModelTestCase):
    """Test the MuRE model."""

    cls = pykeen.models.MuRE
    num_constant_init = 2  # biases


class TestKG2EWithEL(cases.BaseKG2ETest):
    """Test the KG2E model with EL similarity."""

    kwargs = {
        'dist_similarity': 'EL',
    }


class TestNTNLowMemory(cases.BaseNTNTest):
    """Test the NTN model with automatic memory optimization."""

    kwargs = {
        'num_slices': 2,
    }

    training_loop_kwargs = {
        'automatic_memory_optimization': True,
    }


class TestNTNHighMemory(cases.BaseNTNTest):
    """Test the NTN model without automatic memory optimization."""

    kwargs = {
        'num_slices': 2,
    }

    training_loop_kwargs = {
        'automatic_memory_optimization': False,
    }


class TestPairRE(cases.ModelTestCase):
    """Test the PairRE model."""

    cls = pykeen.models.PairRE


class TestProjE(cases.ModelTestCase):
    """Test the ProjE model."""

    cls = pykeen.models.ProjE


class TestQuatE(cases.ModelTestCase):
    """Test the QuatE model."""

    cls = pykeen.models.QuatE
    # quaternion have four components
    embedding_dim = 4 * cases.ModelTestCase.embedding_dim


class TestRESCAL(cases.ModelTestCase):
    """Test the RESCAL model."""

    cls = pykeen.models.RESCAL


class TestRGCNBasis(cases.BaseRGCNTest):
    """Test the R-GCN model."""

    kwargs = {
        'interaction': "transe",
        'interaction_kwargs': dict(p=1),
        'decomposition': "bases",
        "decomposition_kwargs": dict(
            num_bases=3,
        ),
    }
    #: one bias per layer
    num_constant_init = 2


class TestRGCNBlock(cases.BaseRGCNTest):
    """Test the R-GCN model with block decomposition."""

    embedding_dim = 6
    kwargs = {
        'interaction': "distmult",
        'decomposition': "block",
        "decomposition_kwargs": dict(
            num_blocks=3,
        ),
        'edge_weighting': "symmetric",
        'use_batch_norm': True,
    }
    #: (scale & bias for BN) * layers
    num_constant_init = 4


class TestRotatE(cases.ModelTestCase):
    """Test the RotatE model."""

    cls = pykeen.models.RotatE

    def _check_constraints(self):
        """Check model constraints.

        Relation embeddings' entries have to have absolute value 1 (i.e. represent a rotation in complex plane)
        """
        relation_abs = (
            self.instance
                .relation_embeddings(indices=None)
                .view(self.factory.num_relations, -1, 2)
                .norm(p=2, dim=-1)
        )
        assert torch.allclose(relation_abs, torch.ones_like(relation_abs))


class TestSimplE(cases.ModelTestCase):
    """Test the SimplE model."""

    cls = pykeen.models.SimplE


class _BaseTestSE(cases.ModelTestCase):
    """Test the Structured Embedding model."""

    cls = pykeen.models.StructuredEmbedding

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        norms = self.instance.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))


class TestSELowMemory(_BaseTestSE):
    """Tests SE with low memory."""

    training_loop_kwargs = {
        'automatic_memory_optimization': True,
    }


class TestSEHighMemory(_BaseTestSE):
    """Tests SE with low memory."""

    training_loop_kwargs = {
        'automatic_memory_optimization': False,
    }


class TestTransD(cases.DistanceModelTestCase):
    """Test the TransD model."""

    cls = pykeen.models.TransD
    kwargs = {
        'relation_dim': 4,
    }

    def _check_constraints(self):
        """Check model constraints.

        Entity and relation embeddings have to have at most unit L2 norm.
        """
        for emb in (self.instance.entity_embeddings, self.instance.relation_embeddings):
            assert all_in_bounds(emb(indices=None).norm(p=2, dim=-1), high=1., a_tol=EPSILON)

    def test_score_hrt_manual(self):
        """Manually test interaction function of TransD."""
        # entity embeddings
        weights = torch.as_tensor(data=[[2., 2.], [4., 4.]], dtype=torch.float)
        entity_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_embeddings._embeddings.weight.data.copy_(weights)
        self.instance.entity_embeddings = entity_embeddings

        projection_weights = torch.as_tensor(data=[[3., 3.], [2., 2.]], dtype=torch.float)
        entity_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_projection_embeddings._embeddings.weight.data.copy_(projection_weights)
        self.instance.entity_projections = entity_projection_embeddings

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[4.], [4.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=1,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.instance.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[5.], [3.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=1,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
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
        self.instance.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[4., 4., 4.], [4., 4., 4.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
        self.assertAlmostEqual(scores.item(), -27, delta=0.01)

        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
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
        self.instance.entity_embeddings = entity_embeddings

        projection_weights = torch.as_tensor(data=[[2., 2., 2.], [2., 2., 2.]], dtype=torch.float)
        entity_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        entity_projection_embeddings._embeddings.weight.data.copy_(projection_weights)
        self.instance.entity_projections = entity_projection_embeddings

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[3., 3.], [3., 3.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.instance.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[4., 4.], [4., 4.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        second_score = scores[1].item()
        self.assertAlmostEqual(first_score, -18, delta=0.01)
        self.assertAlmostEqual(second_score, -18, delta=0.01)

    def test_project_entity(self):
        """Test _project_entity."""
        # random entity embeddings & projections
        e = torch.rand(1, self.instance.num_entities, self.embedding_dim, generator=self.generator)
        e = clamp_norm(e, maxnorm=1, p=2, dim=-1)
        e_p = torch.rand(1, self.instance.num_entities, self.embedding_dim, generator=self.generator)

        # random relation embeddings & projections
        r = torch.rand(self.batch_size, 1, self.instance.relation_dim, generator=self.generator)
        r = clamp_norm(r, maxnorm=1, p=2, dim=-1)
        r_p = torch.rand(self.batch_size, 1, self.instance.relation_dim, generator=self.generator)

        # project
        e_bot = _project_entity(e=e, e_p=e_p, r=r, r_p=r_p)

        # check shape:
        assert e_bot.shape == (self.batch_size, self.instance.num_entities, self.instance.relation_dim)

        # check normalization
        assert (torch.norm(e_bot, dim=-1, p=2) <= 1.0 + 1.0e-06).all()


class TestTransE(cases.DistanceModelTestCase):
    """Test the TransE model."""

    cls = pykeen.models.TransE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.instance.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransH(cases.DistanceModelTestCase):
    """Test the TransH model."""

    cls = pykeen.models.TransH

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.instance.normal_vector_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransR(cases.DistanceModelTestCase):
    """Test the TransR model."""

    cls = pykeen.models.TransR
    kwargs = {
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
        self.instance.entity_embeddings = entity_embeddings

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[4., 4], [5., 5.]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)
        self.instance.relation_embeddings = relation_embeddings

        relation_projection_weights = torch.as_tensor(data=[[5., 5., 6., 6.], [7., 7., 8., 8.]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=4,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_projections = relation_projection_embeddings

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        # second_score = scores[1].item()
        self.assertAlmostEqual(first_score, -32, delta=0.01)

    def _check_constraints(self):
        """Check model constraints.

        Entity and relation embeddings have to have at most unit L2 norm.
        """
        for emb in (self.instance.entity_embeddings, self.instance.relation_embeddings):
            assert all_in_bounds(emb(indices=None).norm(p=2, dim=-1), high=1., a_tol=1.0e-06)


class TestTuckEr(cases.ModelTestCase):
    """Test the TuckEr model."""

    cls = pykeen.models.TuckER
    kwargs = {
        'relation_dim': 4,
    }
    #: 2xBN (bias & scale)
    num_constant_init = 4


class TestUM(cases.DistanceModelTestCase):
    """Test the Unstructured Model."""

    cls = pykeen.models.UnstructuredModel


class TestCrossE(cases.ModelTestCase):
    """Test the CrossE model."""

    cls = pykeen.models.CrossE

    # the combination bias
    num_constant_init = 1


class TestTesting(unittest_templates.MetaTestCase[Model]):
    """Yo dawg, I heard you like testing, so I wrote a test to test the tests so you can test while you're testing."""

    base_test = cases.ModelTestCase
    base_cls = Model
    skip_cls = SKIP_MODULES

    def test_documentation(self):
        """Test all models have appropriate structured documentation."""
        for name, cls in sorted(model_resolver.lookup_dict.items()):
            with self.subTest(name=name):
                try:
                    docdata = cls.__docdata__
                except AttributeError:
                    self.fail('missing __docdata__')
                self.assertIn('citation', docdata)
                self.assertIn('author', docdata['citation'])
                self.assertIn('link', docdata['citation'])
                self.assertIn('year', docdata['citation'])

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

        star_model_names = _remove_non_models(set(pykeen.models.__all__) - SKIP_MODULES)
        model_names = _remove_non_models(model_names - SKIP_MODULES)

        self.assertEqual(model_names, star_model_names, msg='Forgot to add some imports')

    @unittest.skip('no longer necessary?')
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
            'PairRE',
            'QuatE',
        }
        model_names = _remove_non_models(set(pykeen.models.__all__) - SKIP_MODULES - experiment_blacklist)
        for model in _remove_non_models(model_names):
            with self.subTest(model=model):
                self.assertTrue(
                    os.path.exists(os.path.join(experiments_path, model.lower())),
                    msg=f'Missing experimental configuration for {model}',
                )


def _remove_non_models(elements):
    rv = set()
    for element in elements:
        try:
            model_resolver.lookup(element)
        except ValueError:  # invalid model name - aka not actually a model
            continue
        else:
            rv.add(element)
    return rv


class TestModelUtilities(unittest.TestCase):
    """Extra tests for utility functions."""

    def test_get_novelty_mask(self):
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

    def test_extend_batch(self):
        """Test `_extend_batch()`."""
        batch = torch.tensor([[a, b] for a in range(3) for b in range(4)]).view(-1, 2)
        all_ids = [2 * i for i in range(5)]

        batch_size = batch.shape[0]
        num_choices = len(all_ids)

        for dim in range(3):
            h_ext_batch = extend_batch(batch=batch, all_ids=all_ids, dim=dim)

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


class ERModelTests(cases.ModelTestCase):
    """Tests for the general ER-Model."""

    cls = pykeen.models.ERModel
    kwargs = dict(
        interaction="distmult",  # use name to test interaction resolution
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        embedding_dim = kwargs.pop("embedding_dim")
        kwargs["entity_representations"] = EmbeddingSpecification(embedding_dim=embedding_dim)
        kwargs["relation_representations"] = EmbeddingSpecification(embedding_dim=embedding_dim)
        return kwargs

    def test_has_hpo_defaults(self):  # noqa: D102
        raise unittest.SkipTest(f"Base class {self.cls} does not provide HPO defaults.")
