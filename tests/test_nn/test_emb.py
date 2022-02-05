# -*- coding: utf-8 -*-

"""Test embeddings."""

import unittest
from typing import Any, ClassVar, MutableMapping, Tuple
from unittest.mock import Mock

import numpy
import torch
import unittest_templates

import pykeen.nn.emb
import pykeen.nn.message_passing
import pykeen.nn.node_piece
from pykeen.datasets import get_dataset
from pykeen.triples.generation import generate_triples_factory
from tests import cases, mocks

try:
    import transformers
except ImportError:
    transformers = None


class EmbeddingTests(cases.RepresentationTestCase):
    """Tests for embeddings."""

    cls = pykeen.nn.emb.Embedding
    kwargs = dict(
        num_embeddings=7,
        embedding_dim=13,
    )

    def test_backwards_compatibility(self):
        """Test shape and num_embeddings."""
        assert self.instance.max_id == self.instance.num_embeddings
        embedding_dim = int(numpy.prod(self.instance.shape))
        assert self.instance.shape == (embedding_dim,)

    def test_dropout(self):
        """Test dropout layer."""
        # create a new instance with guaranteed dropout
        kwargs = self.instance_kwargs
        kwargs.pop("dropout", None)
        dropout_instance = self.cls(**kwargs, dropout=0.1)
        # set to training mode
        dropout_instance.train()
        # check for different output
        indices = torch.arange(2)
        first = dropout_instance(indices)
        second = dropout_instance(indices)
        assert not torch.allclose(first, second)


class LowRankEmbeddingRepresentationTests(cases.RepresentationTestCase):
    """Tests for low-rank embedding representations."""

    cls = pykeen.nn.emb.LowRankEmbeddingRepresentation
    kwargs = dict(
        max_id=10,
        shape=(3, 7),
    )


class TensorEmbeddingTests(cases.RepresentationTestCase):
    """Tests for Embedding with 2-dimensional shape."""

    cls = pykeen.nn.emb.Embedding
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

    cls = pykeen.nn.message_passing.RGCNRepresentations
    num_entities: ClassVar[int] = 8
    num_relations: ClassVar[int] = 7
    num_triples: ClassVar[int] = 31
    num_bases: ClassVar[int] = 2
    kwargs = dict(
        embedding_specification=pykeen.nn.emb.EmbeddingSpecification(embedding_dim=num_entities),
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

    cls = pykeen.nn.emb.SingleCompGCNRepresentation
    num_entities: ClassVar[int] = 8
    num_relations: ClassVar[int] = 7
    num_triples: ClassVar[int] = 31
    dim: ClassVar[int] = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["combined"] = pykeen.nn.emb.CombinedCompGCNRepresentations(
            triples_factory=generate_triples_factory(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                num_triples=self.num_triples,
                create_inverse_triples=True,
            ),
            embedding_specification=pykeen.nn.emb.EmbeddingSpecification(embedding_dim=self.dim),
            dims=self.dim,
        )
        return kwargs


class NodePieceRelationTests(cases.NodePieceTestCase):
    """Tests for node piece representation."""

    kwargs = dict(
        token_representations=pykeen.nn.emb.EmbeddingSpecification(
            shape=(3,),
        )
    )


class NodePieceAnchorTests(cases.NodePieceTestCase):
    """Tests for node piece representation with anchor nodes."""

    kwargs = dict(
        token_representations=pykeen.nn.emb.EmbeddingSpecification(
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
        token_representations=(
            pykeen.nn.emb.EmbeddingSpecification(
                shape=(3,),
            ),
            pykeen.nn.emb.EmbeddingSpecification(
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


class TokenizationTests(cases.RepresentationTestCase):
    """Tests for tokenization representation."""

    cls = pykeen.nn.node_piece.TokenizationRepresentationModule
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

    cls = pykeen.nn.emb.SubsetRepresentationModule
    kwargs = dict(
        max_id=7,
    )
    shape: Tuple[int, ...] = (13,)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["base"] = pykeen.nn.emb.Embedding(
            num_embeddings=2 * kwargs["max_id"],
            shape=self.shape,
        )
        return kwargs


@unittest.skipIf(transformers is None, "Need to install `transformers`")
class LabelBasedTransformerRepresentationTests(cases.RepresentationTestCase):
    """Test the label based Transformer representations."""

    cls = pykeen.nn.emb.LabelBasedTransformerRepresentation

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["labels"] = sorted(get_dataset(dataset="nations").entity_to_id.keys())
        return kwargs


class RepresentationModuleMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.emb.RepresentationModule]):
    """Test that there are tests for all representation modules."""

    base_cls = pykeen.nn.emb.RepresentationModule
    base_test = cases.RepresentationTestCase
    skip_cls = {mocks.CustomRepresentations}


class EmbeddingSpecificationTests(unittest.TestCase):
    """Tests for EmbeddingSpecification."""

    #: The number of embeddings
    num: ClassVar[int] = 3

    def test_make(self):
        """Test make."""
        initializer = Mock()
        normalizer = Mock()
        constrainer = Mock()
        regularizer = Mock()
        for embedding_dim, shape in [
            (None, (3,)),
            (None, (3, 5)),
            (3, None),
        ]:
            spec = pykeen.nn.emb.EmbeddingSpecification(
                embedding_dim=embedding_dim,
                shape=shape,
                initializer=initializer,
                normalizer=normalizer,
                constrainer=constrainer,
                regularizer=regularizer,
            )
            emb = spec.make(num_embeddings=self.num)

            # check shape
            self.assertEqual(emb.embedding_dim, (embedding_dim or int(numpy.prod(shape))))
            self.assertEqual(emb.shape, (shape or (embedding_dim,)))
            self.assertEqual(emb.num_embeddings, self.num)

            # check attributes
            self.assertIs(emb.initializer, initializer)
            self.assertIs(emb.normalizer, normalizer)
            self.assertIs(emb.constrainer, constrainer)
            self.assertIs(emb.regularizer, regularizer)

    def test_make_complex(self):
        """Test making a complex embedding."""
        s = pykeen.nn.emb.EmbeddingSpecification(
            shape=(5, 5),
            dtype=torch.cfloat,
        )
        e = s.make(num_embeddings=100)
        self.assertEqual((5, 10), e.shape)

    def test_make_errors(self):
        """Test errors on making with an invalid key."""
        with self.assertRaises(KeyError):
            pykeen.nn.emb.EmbeddingSpecification(
                shape=(1, 1),
                initializer="garbage",
            ).make(num_embeddings=1)
        with self.assertRaises(KeyError):
            pykeen.nn.emb.EmbeddingSpecification(
                shape=(1, 1),
                constrainer="garbage",
            ).make(num_embeddings=1)
        with self.assertRaises(KeyError):
            pykeen.nn.emb.EmbeddingSpecification(
                shape=(1, 1),
                normalizer="garbage",
            ).make(num_embeddings=1)
