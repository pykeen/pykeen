# -*- coding: utf-8 -*-

"""Tests for leakage analysis."""

import itertools as itt
import unittest

import numpy as np

from pykeen.triples import TriplesFactory
from pykeen.triples.leakage import Sealant, get_candidate_inverse_relations


class TestLeakage(unittest.TestCase):
    """Tests for identifying inverse relationships and leakage."""

    def test_count_inverse_frequencies(self):
        """Test counting inverse frequencies.

        Note, for r3, there are three triples, but the inverse triples are only counted once.
        """
        t = [
            ['a', 'r1', 'b'],
            #
            ['b', 'r2', 'c'],
            ['c', 'r2_inverse', 'b'],
            ['d', 'r2', 'e'],
            ['e', 'r2_inverse', 'd'],
            #
            ['g', 'r3', 'h'],
            ['h', 'r3_inverse', 'g'],
            ['i', 'r3', 'j'],
            ['k', 'r3', 'l'],
        ]
        triples_factory = TriplesFactory(triples=np.array(t, dtype=np.str))
        frequencies = get_candidate_inverse_relations(triples_factory, minimum_frequency=0.0, symmetric=False)
        self.assertEqual(
            {
                ('r2', 'r2_inverse'): (2 / 2),
                ('r2_inverse', 'r2'): (2 / 2),
                ('r3', 'r3_inverse'): (1 / 3),
                ('r3_inverse', 'r3'): (1 / 1),
            },
            dict(frequencies),
        )

    def test_find_leak_assymetric(self):
        """Test finding test leakages with an asymmetric metric."""
        n = 100
        test_relation, test_relation_inverse = 'r', 'r_inverse'

        train_generated = list(itt.chain.from_iterable((
            [
                [str(i), test_relation, str(j + 1 + n)],
                [str(j + 1 + n), test_relation_inverse, str(i)],
            ]
            for i, j in zip(range(n), range(n))
        )))
        train_non_inverses = [
            ['a', 'fine', 'b'],
            ['b', 'fine', 'c'],
        ]
        forwards_extras = [
            ['-1', test_relation, '-2'],  # this one leaks!
            ['-3', test_relation, '-4'],
        ]
        inverse_extras = [
            ['-5', test_relation_inverse, '-6'],
        ]
        train = train_generated + train_non_inverses + forwards_extras + inverse_extras
        test = [
            ['-2', test_relation_inverse, '-1'],  # this one was leaked!
        ]
        train_factory = TriplesFactory(triples=np.array(train, dtype=np.str))
        test_factory = TriplesFactory(triples=np.array(test, dtype=np.str))

        sealant = Sealant(train_factory, symmetric=False)

        expected_forwards_frequency = n / (n + len(forwards_extras))
        expected_inverse_frequency = n / (n + len(inverse_extras))
        self.assertGreater(len(forwards_extras), len(inverse_extras))
        self.assertLess(
            expected_forwards_frequency, expected_inverse_frequency,
            msg='Forwards frequency should be higher than inverse frequency',
        )
        self.assertEqual(
            {
                (test_relation, test_relation_inverse): expected_forwards_frequency,
                (test_relation_inverse, test_relation): expected_inverse_frequency,
            },
            dict(sealant.candidate_inverse_relations),
        )

        self.assertIn(test_relation, sealant.inverses)
        self.assertEqual(test_relation_inverse, sealant.inverses[test_relation])
        self.assertIn(test_relation_inverse, sealant.inverses)
        self.assertEqual(test_relation, sealant.inverses[test_relation_inverse])

        self.assertIn(
            test_relation_inverse,
            sealant.inverse_relations_to_delete,
            msg='The wrong relation was picked for deletion',
        )

        test_leaked = sealant.get_inverse_triples(test_factory)
        self.assertEqual(1, len(test_leaked))
        self.assertEqual(('-2', test_relation_inverse, '-1'), tuple(test_leaked[0]))
