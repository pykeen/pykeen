# -*- coding: utf-8 -*-

"""Test embeddings."""

import unittest
from typing import Any, MutableMapping
from unittest.mock import Mock

import numpy
import torch

from pykeen.models.unimodal.rgcn import RGCNRepresentations
from pykeen.nn import Embedding, EmbeddingSpecification, RepresentationModule
from pykeen.triples import TriplesFactory
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
        assert self.instance.shape == (self.instance.embedding_dim,)


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

    cls = RGCNRepresentations
    num = 8
    kwargs = dict(
        num_bases_or_blocks=2,
        embedding_dim=num,
    )
    num_relations: int = 7
    num_triples: int = 31
    num_bases: int = 2

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # TODO: use triple generation
        # generate random triples
        mapped_triples = numpy.stack([
            numpy.random.randint(max_id, size=(self.num_triples,))
            for max_id in (self.num, self.num_relations, self.num)
        ], axis=-1)
        entity_names = [f"e_{i}" for i in range(self.num)]
        relation_names = [f"r_{i}" for i in range(self.num_relations)]
        triples = numpy.stack([
            [names[i] for i in col.tolist()]
            for col, names in zip(
                mapped_triples.T,
                (entity_names, relation_names, entity_names),
            )
        ])
        kwargs["triples_factory"] = TriplesFactory.from_labeled_triples(triples=triples)
        return kwargs


class RepresentationModuleTestsTestCase(cases.TestsTestCase[RepresentationModule]):
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
