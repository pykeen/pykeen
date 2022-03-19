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
from pykeen.triples.generation import generate_triples_factory
from tests import cases, mocks

try:
    import transformers
except ImportError:
    transformers = None


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


# TODO consider making subclass of cases.RepresentationTestCase
# that has num_entities, num_relations, num_triples, and
# create_inverse_triples as well as a generate_triples_factory()
# wrapper


class RGCNRepresentationTests(cases.RepresentationTestCase):
    """Test RGCN representations."""

    cls = pykeen.nn.message_passing.RGCNRepresentation
    num_entities: ClassVar[int] = 8
    num_relations: ClassVar[int] = 7
    num_triples: ClassVar[int] = 31
    num_bases: ClassVar[int] = 2
    kwargs = dict(
        entity_representations_kwargs=dict(embedding_dim=num_entities),
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = generate_triples_factory(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_triples=self.num_triples,
        )
        return kwargs


class TestSingleCompGCNRepresentationTests(cases.RepresentationTestCase):
    """Test single CompGCN representations."""

    cls = pykeen.nn.representation.SingleCompGCNRepresentation
    num_entities: ClassVar[int] = 8
    num_relations: ClassVar[int] = 7
    num_triples: ClassVar[int] = 31
    dim: ClassVar[int] = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["combined"] = pykeen.nn.representation.CombinedCompGCNRepresentations(
            triples_factory=generate_triples_factory(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                num_triples=self.num_triples,
                create_inverse_triples=True,
            ),
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


class RepresentationModuleMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.representation.Representation]):
    """Test that there are tests for all representation modules."""

    base_cls = pykeen.nn.representation.Representation
    base_test = cases.RepresentationTestCase
    skip_cls = {mocks.CustomRepresentation}
