# -*- coding: utf-8 -*-

"""Test embeddings."""

import unittest
from typing import Any, ClassVar, MutableMapping, Tuple

import numpy
import torch
import unittest_templates

import pykeen.nn.message_passing
import pykeen.nn.node_piece
import pykeen.nn.representation
from pykeen.datasets import get_dataset
from tests import cases, mocks

try:
    import transformers
except ImportError:
    transformers = None
error = pykeen.nn.pyg.try_import()


class EmbeddingTests(cases.RepresentationTestCase):
    """Tests for embeddings."""

    cls = pykeen.nn.representation.Embedding
    kwargs = dict(
        num_embeddings=7,
        embedding_dim=13,
    )

    def test_backwards_compatibility(self):
        """Test shape and num_embeddings."""
        assert self.instance.max_id == self.instance.num_embeddings
        embedding_dim = int(numpy.prod(self.instance.shape))
        assert self.instance.shape == (embedding_dim,)


class LowRankEmbeddingRepresentationTests(cases.RepresentationTestCase):
    """Tests for low-rank embedding representations."""

    cls = pykeen.nn.representation.LowRankRepresentation
    kwargs = dict(
        max_id=10,
        shape=(3, 7),
    )


class TensorEmbeddingTests(cases.RepresentationTestCase):
    """Tests for Embedding with 2-dimensional shape."""

    cls = pykeen.nn.representation.Embedding
    kwargs = dict(
        num_embeddings=10,
        shape=(3, 7),
    )


class RGCNRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test RGCN representations."""

    cls = pykeen.nn.message_passing.RGCNRepresentation
    num_bases: ClassVar[int] = 2

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["entity_representations_kwargs"] = (dict(embedding_dim=self.num_entities),)
        return kwargs


class TestSingleCompGCNRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test single CompGCN representations."""

    cls = pykeen.nn.representation.SingleCompGCNRepresentation
    dim: ClassVar[int] = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["combined"] = pykeen.nn.representation.CombinedCompGCNRepresentations(
            triples_factory=kwargs.pop("triples_factory"),
            entity_representations_kwargs=dict(embedding_dim=self.dim),
            relation_representations_kwargs=dict(embedding_dim=self.dim),
            dims=self.dim,
        )
        return kwargs


class NodePieceRelationTests(cases.NodePieceTestCase):
    """Tests for node piece representation."""

    kwargs = dict(
        token_representations_kwargs=dict(
            shape=(3,),
        )
    )


class NodePieceAnchorTests(cases.NodePieceTestCase):
    """Tests for node piece representation with anchor nodes."""

    kwargs = dict(
        token_representations_kwargs=dict(
            shape=(3,),
        ),
        tokenizers="anchor",
        tokenizers_kwargs=dict(
            selection="degree",
        ),
    )


class NodePieceMixedTests(cases.NodePieceTestCase):
    """Tests for node piece representation with mixed tokenizers."""

    kwargs = dict(
        token_representations_kwargs=(
            dict(
                shape=(3,),
            ),
            dict(
                shape=(3,),
            ),
        ),
        tokenizers=("relation", "anchor"),
        num_tokens=(2, 3),
        tokenizers_kwargs=(
            dict(),
            dict(
                selection="degree",
            ),
        ),
    )

    def test_token_representations(self):
        """Verify that the number of token representations is correct."""
        assert len(self.instance.token_representations) == 2


class TokenizationTests(cases.RepresentationTestCase):
    """Tests for tokenization representation."""

    cls = pykeen.nn.node_piece.TokenizationRepresentation
    max_id: int = 13
    vocabulary_size: int = 5
    num_tokens: int = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["assignment"] = torch.randint(self.vocabulary_size, size=(self.max_id, self.num_tokens))
        kwargs["token_representation_kwargs"] = dict(shape=(self.vocabulary_size,))
        return kwargs


class SubsetRepresentationTests(cases.RepresentationTestCase):
    """Tests for subset representations."""

    cls = pykeen.nn.representation.SubsetRepresentation
    kwargs = dict(
        max_id=7,
    )
    shape: Tuple[int, ...] = (13,)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["base"] = pykeen.nn.representation.Embedding(
            num_embeddings=2 * kwargs["max_id"],
            shape=self.shape,
        )
        return kwargs


@unittest.skipIf(transformers is None, "Need to install `transformers`")
class LabelBasedTransformerRepresentationTests(cases.RepresentationTestCase):
    """Test the label based Transformer representations."""

    cls = pykeen.nn.representation.LabelBasedTransformerRepresentation

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["labels"] = sorted(get_dataset(dataset="nations").entity_to_id.keys())
        return kwargs


@unittest.skipIf(error is not None, "Need to install `torch_geometric`")
class IgnoreRelationTypePyGRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test for Pytorch Geometric representations using uni-relational message passing layers."""

    cls = pykeen.nn.pyg.IgnoreRelationTypePyGRepresentation
    embedding_dim: int = 3
    kwargs = dict(
        base_kwargs=dict(shape=(embedding_dim,)),
        layers=["gcn"] * 2,
        layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
    )


@unittest.skipIf(error is not None, "Need to install `torch_geometric`")
class CategoricalRelationTypePyGRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test for Pytorch Geometric representations using categorical message passing layers."""

    cls = pykeen.nn.pyg.CategoricalRelationTypePyGRepresentation
    embedding_dim: int = 3
    kwargs = dict(
        base_kwargs=dict(shape=(embedding_dim,)),
        layers=["rgcn"],
        layers_kwargs=dict(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            num_bases=2,
            num_relations=cases.TriplesFactoryRepresentationTestCase.num_relations,
        ),
    )


@unittest.skipIf(error is not None, "Need to install `torch_geometric`")
class CategoricalRelationTypePyGRepresentationTests(cases.TriplesFactoryRepresentationTestCase):
    """Test for Pytorch Geometric representations using categorical message passing layers."""

    cls = pykeen.nn.pyg.FeaturizedRelationTypePyGRepresentation
    embedding_dim: int = 3
    kwargs = dict(
        base_kwargs=dict(shape=(embedding_dim,)),
        layers=["gat"],
        layers_kwargs=dict(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            num_bases=2,
            num_relations=cases.TriplesFactoryRepresentationTestCase.num_relations,
        ),
    )


class RepresentationModuleMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.representation.Representation]):
    """Test that there are tests for all representation modules."""

    base_cls = pykeen.nn.representation.Representation
    base_test = cases.RepresentationTestCase
    skip_cls = {mocks.CustomRepresentation}
