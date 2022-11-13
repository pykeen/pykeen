# -*- coding: utf-8 -*-

"""Test that models can be executed."""

import importlib
import os
import unittest
from typing import Any, Iterable, MutableMapping, Set, Type, Union

import torch
import unittest_templates

import pykeen.experiments
import pykeen.models
from pykeen.models import (
    ERModel,
    EvaluationOnlyModel,
    FixedModel,
    InductiveERModel,
    Model,
    _NewAbstractModel,
    model_resolver,
)
from pykeen.models.multimodal.base import LiteralModel
from pykeen.nn import Embedding, NodePieceRepresentation
from pykeen.nn.combination import ConcatAggregationCombination
from pykeen.nn.perceptron import ConcatMLP
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.utils import all_in_bounds, extend_batch
from tests import cases
from tests.constants import EPSILON

SKIP_MODULES = {
    Model,
    _NewAbstractModel,
    # DummyModel,
    LiteralModel,
    ERModel,
    InductiveERModel,
    FixedModel,
    EvaluationOnlyModel,
}
SKIP_MODULES.update(LiteralModel.__subclasses__())
SKIP_MODULES.update(EvaluationOnlyModel.__subclasses__())


class TestCompGCN(cases.ModelTestCase):
    """Test the CompGCN model."""

    cls = pykeen.models.CompGCN
    create_inverse_triples = True
    num_constant_init = 3  # BN(2) + Bias
    cli_extras = ["--create-inverse-triples"]

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        dim = kwargs.pop("embedding_dim")
        kwargs["encoder_kwargs"] = dict(
            entity_representations_kwargs=dict(
                shape=(dim,),
            ),
            relation_representations_kwargs=dict(
                shape=(dim,),
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
        "output_channels": 2,
        "embedding_height": 3,
        "embedding_width": 4,
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
        "num_filters": 2,
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
        entity_norms = self.instance.entity_representations[0](indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestDistMA(cases.ModelTestCase):
    """Test the DistMA model."""

    cls = pykeen.models.DistMA


class TestERMLP(cases.ModelTestCase):
    """Test the ERMLP model."""

    cls = pykeen.models.ERMLP
    kwargs = {
        "hidden_dim": 4,
    }
    # Two linear layer biases
    num_constant_init = 2


class TestERMLPE(cases.ModelTestCase):
    """Test the extended ERMLP model."""

    cls = pykeen.models.ERMLPE
    kwargs = {
        "hidden_dim": 4,
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
        assert all_in_bounds(
            self.instance.entity_representations[0](indices=None).norm(p=2, dim=-1), high=1.0, a_tol=EPSILON
        )


class TestKG2EWithKL(cases.BaseKG2ETest):
    """Test the KG2E model with KL similarity."""

    kwargs = {
        "dist_similarity": "KL",
    }


class TestMuRE(cases.ModelTestCase):
    """Test the MuRE model."""

    cls = pykeen.models.MuRE
    num_constant_init = 2  # biases


class TestKG2EWithEL(cases.BaseKG2ETest):
    """Test the KG2E model with EL similarity."""

    kwargs = {
        "dist_similarity": "EL",
    }


class TestNodePiece(cases.BaseNodePieceTest):
    """Test the NodePiece model."""

    def test_disconnected(self):
        """Test handling of disconnected entities."""
        edges = torch.tensor(
            [[0, 0, 1], [1, 1, 0], [3, 1, 0], [3, 2, 1]], dtype=torch.long
        )  # node ID 2 is missing as a disconnected node
        factory = CoreTriplesFactory.create(
            mapped_triples=edges, num_entities=4, num_relations=3, create_inverse_triples=True
        )
        pykeen.models.NodePiece(triples_factory=factory, num_tokens=2)


class TestNodePieceMLP(cases.BaseNodePieceTest):
    """Test the NodePiece model with MLP aggregation."""

    kwargs = dict(aggregation="mlp")

    def test_aggregation(self):
        """Test that the MLP gets registered properly and is trainable."""
        self.assertIsInstance(self.instance, pykeen.models.NodePiece)
        r = self.instance.entity_representations[0]
        self.assertIsInstance(r, NodePieceRepresentation)
        self.assertIsInstance(r.combination, ConcatAggregationCombination)
        self.assertIsInstance(r.combination.aggregation, ConcatMLP)

        # Test that the weight in the MLP is trainable (i.e. requires grad)
        for key in [
            f"entity_representations.0.combination.aggregation.{key}"
            for key in ("0.weight", "0.bias", "3.weight", "3.bias")
        ]:
            params = dict(self.instance.named_parameters())
            self.assertIn(key, set(params))
            tensor = params[key]
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertTrue(tensor.requires_grad)


class TestNodePieceAnchors(cases.BaseNodePieceTest):
    """Test the NodePiece model with anchors."""

    kwargs = dict(
        tokenizers="anchor",
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["tokenizers_kwargs"] = dict(selection_kwargs=dict(num_anchors=self.factory.num_entities // 3))
        return kwargs


class TestNodePieceJoint(cases.BaseNodePieceTest):
    """Test the NodePiece model with joint anchor and relation tokenization."""

    num_anchors = 5
    num_tokens = [3, 2]
    kwargs = dict(
        tokenizers=["anchor", "relation"],
        tokenizers_kwargs=[
            dict(
                selection="degree",
                searcher="scipy-sparse",
            ),
            dict(),
        ],
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["num_tokens"] = self.num_tokens
        kwargs["tokenizers_kwargs"][0]["selection_kwargs"] = dict(num_anchors=self.num_anchors)
        return kwargs

    def test_vocabulary_size(self):
        """Test the expected vocabulary size of the individual tokenizations."""
        node_piece = self.instance.entity_representations[0]
        assert isinstance(node_piece, NodePieceRepresentation)
        assert isinstance(node_piece.base, torch.nn.ModuleList)
        assert len(node_piece.base) == 2
        anchor, relation = node_piece.base
        assert anchor.vocabulary.max_id == self.num_anchors + 1
        assert relation.vocabulary.max_id == 2 * self.factory.real_num_relations + 1


class TestInductiveNodePiece(cases.InductiveModelTestCase):
    """Test the InductiveNodePiece model."""

    cls = pykeen.models.InductiveNodePiece
    create_inverse_triples = True


class TestInductiveNodePieceGNN(cases.InductiveModelTestCase):
    """Test the InductiveNodePieceGNN model."""

    cls = pykeen.models.InductiveNodePieceGNN
    num_constant_init = 6
    create_inverse_triples = True
    train_batch_size = 8


class TestNTN(cases.ModelTestCase):
    """Test the NTN model."""

    cls = pykeen.models.NTN

    kwargs = {
        "num_slices": 2,
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
        "interaction": "transe",
        "interaction_kwargs": dict(p=1),
        "decomposition": "bases",
        "decomposition_kwargs": dict(
            num_bases=3,
        ),
    }


class TestRGCNBlock(cases.BaseRGCNTest):
    """Test the R-GCN model with block decomposition."""

    embedding_dim = 6
    kwargs = {
        "interaction": "distmult",
        "decomposition": "block",
        "decomposition_kwargs": dict(
            num_blocks=3,
        ),
        "edge_weighting": "symmetric",
    }


class TestRotatE(cases.ModelTestCase):
    """Test the RotatE model."""

    cls = pykeen.models.RotatE

    def _check_constraints(self):
        """Check model constraints.

        Relation embeddings' entries have to have absolute value 1 (i.e. represent a rotation in complex plane)
        """
        relation_abs = self.instance.relation_representations[0](indices=None).abs()
        assert torch.allclose(relation_abs, torch.ones_like(relation_abs))


class TestSimplE(cases.ModelTestCase):
    """Test the SimplE model."""

    cls = pykeen.models.SimplE


class TestSE(cases.ModelTestCase):
    """Test the Structured Embedding model."""

    cls = pykeen.models.SE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        norms = self.instance.entity_representations[0](indices=None).norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))


class TestTorusE(cases.DistanceModelTestCase):
    """Test the TorusE model."""

    cls = pykeen.models.TorusE


class TestTransD(cases.DistanceModelTestCase):
    """Test the TransD model."""

    cls = pykeen.models.TransD
    kwargs = {
        "relation_dim": 4,
    }

    def _check_constraints(self):
        """Check model constraints.

        Entity and relation embeddings have to have at most unit L2 norm.
        """
        self.instance: ERModel
        for emb in (self.instance.entity_representations[0], self.instance.relation_representations[0]):
            assert all_in_bounds(emb(indices=None).norm(p=2, dim=-1), high=1.0, a_tol=EPSILON)

    def test_score_hrt_manual(self):
        """Manually test interaction function of TransD."""
        # entity embeddings
        weights = torch.as_tensor(data=[[2.0, 2.0], [4.0, 4.0]], dtype=torch.float)
        entity_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_embeddings._embeddings.weight.data.copy_(weights)

        projection_weights = torch.as_tensor(data=[[3.0, 3.0], [2.0, 2.0]], dtype=torch.float)
        entity_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        entity_projection_embeddings._embeddings.weight.data.copy_(projection_weights)
        self.instance.entity_representations = torch.nn.ModuleList([entity_embeddings, entity_projection_embeddings])

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[4.0], [4.0]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=1,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)

        relation_projection_weights = torch.as_tensor(data=[[5.0], [3.0]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=1,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_representations = torch.nn.ModuleList(
            [relation_embeddings, relation_projection_embeddings]
        )

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        self.assertAlmostEqual(first_score, -16, delta=0.01)

        # Use different dimension for relation embedding: relation_dim > entity_dim
        # relation embeddings
        relation_weights = torch.as_tensor(data=[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)

        relation_projection_weights = torch.as_tensor(data=[[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_representations = torch.nn.ModuleList(
            [relation_embeddings, relation_projection_embeddings]
        )

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
        weights = torch.as_tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float)
        entity_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        entity_embeddings._embeddings.weight.data.copy_(weights)

        projection_weights = torch.as_tensor(data=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float)
        entity_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=3,
        )
        entity_projection_embeddings._embeddings.weight.data.copy_(projection_weights)
        self.instance.entity_representations = torch.nn.ModuleList([entity_embeddings, entity_projection_embeddings])

        # relation embeddings
        relation_weights = torch.as_tensor(data=[[3.0, 3.0], [3.0, 3.0]], dtype=torch.float)
        relation_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_embeddings._embeddings.weight.data.copy_(relation_weights)

        relation_projection_weights = torch.as_tensor(data=[[4.0, 4.0], [4.0, 4.0]], dtype=torch.float)
        relation_projection_embeddings = Embedding(
            num_embeddings=2,
            embedding_dim=2,
        )
        relation_projection_embeddings._embeddings.weight.data.copy_(relation_projection_weights)
        self.instance.relation_representations = torch.nn.ModuleList(
            [relation_embeddings, relation_projection_embeddings]
        )

        # Compute Scores
        batch = torch.as_tensor(data=[[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        scores = self.instance.score_hrt(hrt_batch=batch)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 1)
        first_score = scores[0].item()
        second_score = scores[1].item()
        self.assertAlmostEqual(first_score, -18, delta=0.01)
        self.assertAlmostEqual(second_score, -18, delta=0.01)


class TestTransE(cases.DistanceModelTestCase):
    """Test the TransE model."""

    cls = pykeen.models.TransE

    def _check_constraints(self):
        """Check model constraints.

        Entity embeddings have to have unit L2 norm.
        """
        entity_norms = self.instance.entity_representations[0](indices=None).norm(p=2, dim=-1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms))


class TestTransF(cases.ModelTestCase):
    """Test the TransF model."""

    cls = pykeen.models.TransF


class TestTransH(cases.DistanceModelTestCase):
    """Test the TransH model."""

    cls = pykeen.models.TransH

    def _check_constraints(self):
        """Check model constraints.

        Normal vectors of relation-specific hyperplanes have unit length.
        """
        norm = self.instance.relation_representations[1](indices=None).norm(p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))


class TestTransR(cases.DistanceModelTestCase):
    """Test the TransR model."""

    cls = pykeen.models.TransR
    kwargs = {
        "relation_dim": 4,
    }

    def _check_constraints(self):
        """Check model constraints.

        Entity and relation embeddings have to have at most unit L2 norm.
        """
        for emb in (self.instance.entity_representations[0], self.instance.relation_representations[0]):
            assert all_in_bounds(emb(indices=None).norm(p=2, dim=-1), high=1.0, a_tol=1.0e-06)


class TestTuckEr(cases.ModelTestCase):
    """Test the TuckEr model."""

    cls = pykeen.models.TuckER
    kwargs = {
        "relation_dim": 4,
    }
    #: 2xBN (bias & scale)
    num_constant_init = 4


class TestUM(cases.DistanceModelTestCase):
    """Test the Unstructured Model."""

    cls = pykeen.models.UM


class TestCrossE(cases.ModelTestCase):
    """Test the CrossE model."""

    cls = pykeen.models.CrossE

    # the combination bias
    num_constant_init = 1


class TestBoxE(cases.ModelTestCase):
    """Test the BoxE model."""

    cls = pykeen.models.BoxE


class TestCP(cases.ModelTestCase):
    """Test the CP model."""

    cls = pykeen.models.CP


class TestAutoSF(cases.ModelTestCase):
    """Test the AutoSF model."""

    cls = pykeen.models.AutoSF


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
                    self.fail("missing __docdata__")
                self.assertIn("citation", docdata)
                self.assertIn("author", docdata["citation"])
                self.assertIn("link", docdata["citation"])
                self.assertIn("year", docdata["citation"])

    def test_importing(self):
        """Test that all models are available from :mod:`pykeen.models`."""
        models_path = os.path.abspath(os.path.dirname(pykeen.models.__file__))

        model_names = set()
        for directory, _, filenames in os.walk(models_path):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue

                path = os.path.join(directory, filename)
                relpath = os.path.relpath(path, models_path)
                if relpath.endswith("__init__.py"):
                    continue

                import_path = "pykeen.models." + relpath[: -len(".py")].replace(os.sep, ".")
                module = importlib.import_module(import_path)

                for name in dir(module):
                    value = getattr(module, name)
                    if isinstance(value, type) and issubclass(value, Model):
                        model_names.add(value.__name__)

        star_model_names = _remove_non_models(set(pykeen.models.__all__) - SKIP_MODULES)
        # FIXME definitely a type mismatch going on here
        model_names = _remove_non_models(model_names - SKIP_MODULES)

        self.assertEqual(model_names, star_model_names, msg="Forgot to add some imports")

    @unittest.skip("no longer necessary?")
    def test_models_have_experiments(self):
        """Test that each model has an experiment folder in :mod:`pykeen.experiments`."""
        experiments_path = os.path.abspath(os.path.dirname(pykeen.experiments.__file__))
        experiment_blacklist = {
            "DistMultLiteral",  # FIXME
            "ComplExLiteral",  # FIXME
            "UnstructuredModel",
            "StructuredEmbedding",
            "RESCAL",
            "NTN",
            "ERMLP",
            "ProjE",  # FIXME
            "ERMLPE",  # FIXME
            "PairRE",
            "QuatE",
        }
        model_names = _remove_non_models(set(pykeen.models.__all__) - SKIP_MODULES - experiment_blacklist)
        for model in _remove_non_models(model_names):
            with self.subTest(model=model):
                self.assertTrue(
                    os.path.exists(os.path.join(experiments_path, model.lower())),
                    msg=f"Missing experimental configuration for {model}",
                )


def _remove_non_models(elements: Iterable[Union[str, Type[Model]]]) -> Set[Type[Model]]:
    rv = set()
    for element in elements:
        try:
            model_cls = model_resolver.lookup(element)
        except KeyError:  # invalid model name - aka not actually a model
            continue
        else:
            rv.add(model_cls)
    return rv


class TestModelUtilities(unittest.TestCase):
    """Extra tests for utility functions."""

    def test_extend_batch(self):
        """Test `_extend_batch()`."""
        batch = torch.tensor([[a, b] for a in range(3) for b in range(4)]).view(-1, 2)
        max_id = 5

        batch_size = batch.shape[0]
        num_choices = max_id

        for dim in range(3):
            h_ext_batch = extend_batch(batch=batch, max_id=max_id, dim=dim)

            # check shape
            assert h_ext_batch.shape == (batch_size * num_choices, 3)

            # check content
            actual_content = set(tuple(map(int, hrt)) for hrt in h_ext_batch)
            exp_content = set()
            for i in range(max_id):
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
        shape = (kwargs.pop("embedding_dim"),)
        kwargs["entity_representations_kwargs"] = dict(shape=shape)
        kwargs["relation_representations_kwargs"] = dict(shape=shape)
        return kwargs

    def test_has_hpo_defaults(self):  # noqa: D102
        raise unittest.SkipTest(f"Base class {self.cls} does not provide HPO defaults.")


class CooccurrenceFilteredModelTests(cases.ModelTestCase):
    """Tests for the filtered meta model."""

    cls = pykeen.models.meta.filtered.CooccurrenceFilteredModel
