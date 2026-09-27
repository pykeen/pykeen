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
            assert reference.training.num_triples == derived.training.num_triples
            assert not (reference.training.mapped_triples == derived.training.mapped_triples).all()

            assert reference.testing.num_triples == derived.testing.num_triples
            assert not (reference.testing.mapped_triples == derived.testing.mapped_triples).all()

            assert reference.validation.num_triples == derived.validation.num_triples
            assert not (reference.validation.mapped_triples == derived.validation.mapped_triples).all()
