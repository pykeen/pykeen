# -*- coding: utf-8 -*-

"""Tests for the remix algorithms."""

import unittest

from pykeen.datasets import Nations


class TestRemix(unittest.TestCase):
    """Tests for the remix algorithm."""

    def test_remix(self):
        """Test the remix algorithm."""
        reference = Nations()
        for random_state in range(20):
            derived = reference.remix(random_state=random_state)
            self.assertEqual(reference.training.num_triples, derived.training.num_triples)
            self.assertFalse((reference.training.mapped_triples == derived.training.mapped_triples).all())

            self.assertEqual(reference.testing.num_triples, derived.testing.num_triples)
            self.assertFalse((reference.testing.mapped_triples == derived.testing.mapped_triples).all())

            self.assertEqual(reference.validation.num_triples, derived.validation.num_triples)
            self.assertFalse((reference.validation.mapped_triples == derived.validation.mapped_triples).all())
