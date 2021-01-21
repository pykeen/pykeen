# -*- coding: utf-8 -*-

"""Tests for generating triples."""

import unittest
from typing import Set

from pykeen.triples.generation import generate_triples, generate_triples_factory
from pykeen.triples.utils import get_entities, get_relations


class TestGenerate(unittest.TestCase):
    """Tests for generation of triples."""

    num_entities: int = 33
    num_relations: int = 7
    num_triples: int = 101

    def assert_consecutive(self, x: Set[int], msg=None):
        """Assert that all of the things in the collection are consecutive integers."""
        self.assertEqual(set(range(len(x))), x, msg=msg)

    def test_compacted(self):
        """Test that the results are compacted."""
        for random_state in range(100):
            x = generate_triples(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                num_triples=self.num_triples,
                random_state=random_state,
            )
            self.assertEqual(self.num_triples, x.shape[0])
            self.assert_consecutive(get_entities(x))
            self.assert_consecutive(get_relations(x))

    def test_generate_triples_factory(self):
        """Test generating a triples factory."""
        for random_state in range(100):
            tf = generate_triples_factory(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                num_triples=self.num_triples,
                random_state=random_state,
            )
            self.assertEqual(self.num_triples, tf.mapped_triples.shape[0])
            self.assert_consecutive(get_entities(tf.mapped_triples))
            self.assert_consecutive(get_relations(tf.mapped_triples))
