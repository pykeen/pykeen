# -*- coding: utf-8 -*-

"""Tests for leakage analysis."""

import itertools as itt
import unittest

import numpy as np
import torch

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
        triples_factory = TriplesFactory.from_labeled_triples(triples=np.array(t, dtype=np.str))
        frequencies = get_candidate_inverse_relations(triples_factory, minimum_frequency=0.0, symmetric=False)
        self.assertEqual(
            {
                (triples_factory.relation_to_id['r2'], triples_factory.relation_to_id['r2_inverse']): (2 / 2),
                (triples_factory.relation_to_id['r2_inverse'], triples_factory.relation_to_id['r2']): (2 / 2),
                (triples_factory.relation_to_id['r3'], triples_factory.relation_to_id['r3_inverse']): (1 / 3),
                (triples_factory.relation_to_id['r3_inverse'], triples_factory.relation_to_id['r3']): (1 / 1),
            },
            dict(frequencies),
        )

    def test_find_leak_asymmetric(self):
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
        train_factory = TriplesFactory.from_labeled_triples(triples=np.array(train, dtype=np.str))
        test_factory = TriplesFactory.from_labeled_triples(
            triples=np.array(test, dtype=np.str),
            entity_to_id=train_factory.entity_to_id,
            relation_to_id=train_factory.relation_to_id,
        )

        sealant = Sealant(train_factory, symmetric=False)

        expected_forwards_frequency = n / (n + len(forwards_extras))
        expected_inverse_frequency = n / (n + len(inverse_extras))
        self.assertGreater(len(forwards_extras), len(inverse_extras))
        self.assertLess(
            expected_forwards_frequency, expected_inverse_frequency,
            msg='Forwards frequency should be higher than inverse frequency',
        )
        test_relation_id = train_factory.relation_to_id[test_relation]
        test_relation_inverse_id = train_factory.relation_to_id[test_relation_inverse]
        self.assertEqual(
            {
                (test_relation_id, test_relation_inverse_id): expected_forwards_frequency,
                (test_relation_inverse_id, test_relation_id): expected_inverse_frequency,
            },
            dict(sealant.candidate_inverse_relations),
        )

        self.assertIn(test_relation_id, sealant.inverses)
        self.assertEqual(test_relation_inverse_id, sealant.inverses[test_relation_id])
        self.assertIn(test_relation_inverse_id, sealant.inverses)
        self.assertEqual(test_relation_id, sealant.inverses[test_relation_inverse_id])

        self.assertIn(
            test_relation_inverse_id,
            sealant.inverse_relations_to_delete,
            msg='The wrong relation was picked for deletion',
        )

        test_leaked = sealant.get_inverse_triples(test_factory)
        self.assertEqual(1, test_leaked.shape[0])
        assert (torch.as_tensor(data=[
            train_factory.entity_to_id["-2"],
            test_relation_inverse_id,
            train_factory.entity_to_id["-1"],
        ]) == test_leaked[0]).all()
