# -*- coding: utf-8 -*-

"""Test embeddings."""

import unittest
from typing import Any, MutableMapping
from unittest.mock import Mock

import numpy
import torch
import unittest_templates

import pykeen.nn.emb
from pykeen.datasets.nations import NationsLiteral
from pykeen.nn.emb import Embedding, EmbeddingSpecification, LiteralRepresentation, RepresentationModule
from pykeen.triples.generation import generate_triples_factory
from tests import cases, mocks


class EmbeddingTests(cases.RepresentationTestCase):
    """Tests for embeddings."""

    cls = Embedding
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


class LiteralEmbeddingTests(cases.RepresentationTestCase):
    """Tests for literal embeddings."""

    cls = LiteralRepresentation
    kwargs = dict(
        numeric_literals=NationsLiteral().training.numeric_literals,
    )


class TensorEmbeddingTests(cases.RepresentationTestCase):
    """Tests for Embedding with 2-dimensional shape."""

    cls = Embedding
    exp_shape = (3, 7)
    kwargs = dict(
        num_embeddings=10,
        shape=(3, 7),
    )


class RGCNRepresentationTests(cases.RepresentationTestCase):
    """Test RGCN representations."""

    cls = pykeen.nn.emb.RGCNRepresentations
    num = 8
    kwargs = dict(
        embedding_specification=EmbeddingSpecification(embedding_dim=num),
    )
    num_relations: int = 7
    num_triples: int = 31
    num_bases: int = 2

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = generate_triples_factory(
            num_entities=self.num,
            num_relations=self.num_relations,
            num_triples=self.num_triples,
        )
        return kwargs


class TestSingleCompGCNRepresentationTests(cases.RepresentationTestCase):
    """Test single CompGCN representations."""

    cls = pykeen.nn.emb.SingleCompGCNRepresentation
    num_entities: int = 8
    num_relations: int = 7
    num_triples: int = 31
    dim: int = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["combined"] = pykeen.nn.emb.CombinedCompGCNRepresentations(
            triples_factory=generate_triples_factory(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                num_triples=self.num_triples,
                create_inverse_triples=True,
            ),
            embedding_specification=EmbeddingSpecification(embedding_dim=self.dim),
            dims=self.dim,
        )
        return kwargs


class RepresentationModuleTestsTestCase(unittest_templates.MetaTestCase[RepresentationModule]):
    """Test that there are tests for all representation modules."""

    base_cls = RepresentationModule
    base_test = cases.RepresentationTestCase
    skip_cls = {mocks.CustomRepresentations}


class EmbeddingSpecificationTests(unittest.TestCase):
    """Tests for EmbeddingSpecification."""

    #: The number of embeddings
    num: int = 3

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
            spec = EmbeddingSpecification(
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
        s = EmbeddingSpecification(
            shape=(5, 5),
            dtype=torch.cfloat,
        )
        e = s.make(num_embeddings=100)
        self.assertEqual((5, 10), e.shape)

    def test_make_errors(self):
        """Test errors on making with an invalid key."""
        with self.assertRaises(KeyError):
            EmbeddingSpecification(
                shape=(1, 1),
                initializer='garbage',
            ).make(num_embeddings=1)
        with self.assertRaises(KeyError):
            EmbeddingSpecification(
                shape=(1, 1),
                constrainer='garbage',
            ).make(num_embeddings=1)
        with self.assertRaises(KeyError):
            EmbeddingSpecification(
                shape=(1, 1),
                normalizer='garbage',
            ).make(num_embeddings=1)
