# -*- coding: utf-8 -*-

"""Tests for deterioration workflow."""

import unittest

from pykeen.datasets.base import Dataset
from pykeen.datasets.nations import Nations
from pykeen.triples.triples_factory import splits_steps
from pykeen.utils import ensure_torch_random_state


class TestDeterioration(unittest.TestCase):
    """Tests for deterioration workflow."""

    def setUp(self) -> None:
        """Set up deterioration tests."""
        self.generator = ensure_torch_random_state(42)
        self.reference = Nations()
        self.num_training_triples = self.reference.training.num_triples
        self.num_triples = (
            self.reference.training.num_triples
            + self.reference.testing.num_triples
            + self.reference.validation.num_triples
        )

    def test_deteriorate(self):
        """Test deterioration on integer values for ``n``."""
        for n in [1, 2, 5, 10, 50, 100, 500, 1000]:
            with self.subTest(n=n):
                derived = self.reference.deteriorate(n=n, random_state=self.generator)
                self._help_check(derived)
                self.assertEqual(n, splits_steps(self.reference._tup(), derived._tup()))
                self.assertEqual(1 - n / self.num_triples, self.reference.similarity(derived), msg='similarity')
                self.assertEqual(1 - n / self.num_triples, derived.similarity(self.reference), msg='similarity')

    def test_deteriorate_frac(self):
        """Test deterioration on fractional values for ``n``."""
        for n_frac in [
            1 / self.num_training_triples,
            2 / self.num_training_triples,
            5 / self.num_training_triples,
            0.1,
            0.2,
            0.3,
        ]:
            n = int(n_frac * self.num_training_triples)
            with self.subTest(n=n, n_frac=n_frac):
                derived = self.reference.deteriorate(n=n, random_state=self.generator)
                self._help_check(derived)
                self.assertEqual(
                    n,
                    splits_steps(self.reference._tup(), derived._tup()),
                    msg='steps',
                )
                self.assertEqual(1 - n / self.num_triples, self.reference.similarity(derived), msg='similarity')
                self.assertEqual(1 - n / self.num_triples, derived.similarity(self.reference), msg='similarity')

    def _help_check(self, derived: Dataset):
        self.assertIsNotNone(derived.validation)
        self.assertEqual(self.num_training_triples, self.reference.training.num_triples)
        self.assertEqual(
            self.num_triples,
            sum((
                derived.training.num_triples,
                derived.testing.num_triples,
                derived.validation.num_triples,
            )),
            msg='different number of total triples',
        )
        self.assertLess(derived.training.num_triples, self.reference.training.num_triples)
