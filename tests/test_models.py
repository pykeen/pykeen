# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import unittest
from typing import Optional

import numpy
import torch
from torch import nn

import pykeen.experiments
import pykeen.models
from pykeen.models import (
    EntityEmbeddingModel, EntityRelationEmbeddingModel, Model, MultimodalModel, _MODELS,
    _OldAbstractModel,
)
from pykeen.models.predict import get_novelty_mask, predict
from pykeen.models.unimodal.rgcn import (
    inverse_indegree_edge_weights,
    inverse_outdegree_edge_weights,
    symmetric_edge_weights,
)
from pykeen.models.unimodal.trans_d import _project_entity
from pykeen.nn import Embedding, RepresentationModule
from pykeen.utils import all_in_bounds, clamp_norm, extend_batch
from tests import cases

SKIP_MODULES = {
    Model.__name__,
    _OldAbstractModel.__name__,
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


class TestComplex(cases.ModelTestCase):
    """Test the ComplEx model."""

    model_cls = pykeen.models.ComplEx


class TestConvE(cases.ModelTestCase):
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


class TestConvKB(cases.ModelTestCase):
    """Test the ConvKB model."""

    model_cls = pykeen.models.ConvKB
    model_kwargs = {
        'num_filters': 2,
    }
    # two bias terms, one conv-filter
    num_constant_init = 3


class TestDistMult(cases.ModelTestCase):
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
        top_triples, top_scores = predict(model=self.model, batch_size=batch_size, k=k)

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


class TestERMLP(cases.ModelTestCase):
    """Test the ERMLP model."""

    model_cls = pykeen.models.ERMLP
    model_kwargs = {
        'hidden_dim': 4,
    }
    # Two linear layer biases
    num_constant_init = 2


class TestERMLPE(cases.ModelTestCase):
    """Test the extended ERMLP model."""

    model_cls = pykeen.models.ERMLPE
    model_kwargs = {
        'hidden_dim': 4,
    }
    # Two BN layers, bias & scale
    num_constant_init = 4


class TestHolE(cases.ModelTestCase):
    """Test the HolE model."""

    model_cls = pykeen.models.HolE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have at most unit L2 norm.
        """
        assert all_in_bounds(self.model.entity_embeddings(indices=None).norm(p=2, dim=-1), high=1., a_tol=_EPSILON)


class TestKG2EWithKL(cases.BaseKG2ETest):
    """Test the KG2E model with KL similarity."""

    model_kwargs = {
        'dist_similarity': 'KL',
    }


class TestKG2EWithEL(cases.BaseKG2ETest):
    """Test the KG2E model with EL similarity."""

    model_kwargs = {
        'dist_similarity': 'EL',
    }


class TestNTNLowMemory(cases.BaseNTNTest):
    """Test the NTN model with automatic memory optimization."""

    model_kwargs = {
        'num_slices': 2,
    }

    training_loop_kwargs = {
        'automatic_memory_optimization': True,
    }


class TestNTNHighMemory(cases.BaseNTNTest):
    """Test the NTN model without automatic memory optimization."""

    model_kwargs = {
        'num_slices': 2,
    }

    training_loop_kwargs = {
        'automatic_memory_optimization': False,
    }


class TestProjE(cases.ModelTestCase):
    """Test the ProjE model."""

    model_cls = pykeen.models.ProjE


class TestRESCAL(cases.ModelTestCase):
    """Test the RESCAL model."""

    model_cls = pykeen.models.RESCAL


class TestRGCNBasis(cases.BaseRGCNTest):
    """Test the R-GCN model."""

    model_kwargs = {
        'decomposition': 'basis',
    }
    #: one bias per layer
    num_constant_init = 2


class TestRGCNBlock(cases.BaseRGCNTest):
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


class TestRotatE(cases.ModelTestCase):
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


class TestSimplE(cases.ModelTestCase):
    """Test the SimplE model."""

    model_cls = pykeen.models.SimplE


class _BaseTestSE(cases.ModelTestCase):
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


class TestTransE(cases.DistanceModelTestCase):
    """Test the TransE model."""

    model_cls = pykeen.models.TransE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.model.entity_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransH(cases.DistanceModelTestCase):
    """Test the TransH model."""

    model_cls = pykeen.models.TransH

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.model.normal_vector_embeddings(indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransR(cases.DistanceModelTestCase):
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


class TestTuckEr(cases.ModelTestCase):
    """Test the TuckEr model."""

    model_cls = pykeen.models.TuckER
    model_kwargs = {
        'relation_dim': 4,
    }
    #: 2xBN (bias & scale)
    num_constant_init = 4


class TestUM(cases.DistanceModelTestCase):
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
                and issubclass(value, cases.ModelTestCase)
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


class TestModelUtilities(unittest.TestCase):
    """Extra tests for utility functions."""

    def test_abstract(self):
        """Test that classes are checked as abstract properly."""
        self.assertTrue(EntityEmbeddingModel._is_base_model)
        self.assertTrue(EntityRelationEmbeddingModel._is_base_model)
        self.assertTrue(MultimodalModel._is_base_model)
        for model_cls in _MODELS:
            self.assertFalse(
                model_cls._is_base_model,
                msg=f'{model_cls.__name__} should not be marked as a a base model',
            )

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
