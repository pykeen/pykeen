# -*- coding: utf-8 -*-

"""Tests for leakage analysis."""

import itertools as itt
import unittest

import numpy as np
import torch

from pykeen.triples import TriplesFactory
from pykeen.triples.leakage import Sealant, _generate_compact_vectorized_lookup, get_candidate_inverse_relations


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
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=np.array(t, dtype=np.str),
            filter_out_candidate_inverse_relations=False,
        )
        frequencies = get_candidate_inverse_relations(triples_factory, minimum_frequency=0.0, symmetric=False)
        expected_frequencies = {
            ('r2', 'r2_inverse'): (2 / 2),
            ('r2_inverse', 'r2'): (2 / 2),
            ('r3', 'r3_inverse'): (1 / 3),
            ('r3_inverse', 'r3'): (1 / 1),
        }
        expected_frequencies = {
            (triples_factory.relation_to_id[r1n], triples_factory.relation_to_id[r2n]): count
            for (r1n, r2n), count in expected_frequencies.items()
        }
        self.assertEqual(
            expected_frequencies,
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
        train_factory = TriplesFactory.from_labeled_triples(
            triples=np.array(train, dtype=np.str),
            filter_out_candidate_inverse_relations=False,
        )
        test_factory = TriplesFactory.from_labeled_triples(
            triples=np.array(test, dtype=np.str),
            entity_to_id=train_factory.entity_to_id,
            relation_to_id=train_factory.relation_to_id,
            filter_out_candidate_inverse_relations=False,
        )

        sealant = Sealant(train_factory, symmetric=False)

        expected_forwards_frequency = n / (n + len(forwards_extras))
        expected_inverse_frequency = n / (n + len(inverse_extras))
        self.assertGreater(len(forwards_extras), len(inverse_extras))
        self.assertLess(
            expected_forwards_frequency, expected_inverse_frequency,
            msg='Forwards frequency should be higher than inverse frequency',
        )

        test_relation, test_relation_inverse = [
            train_factory.relation_to_id[r] for r in (test_relation, test_relation_inverse)
        ]
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

        # Test looking up inverse triples
        test_leaked = test_factory.mapped_triples[
            test_factory.get_mask_for_relations(relations=sealant.inverse_relations_to_delete, invert=False)
        ]
        self.assertEqual(1, len(test_leaked))
        self.assertEqual(
            (train_factory.entity_to_id['-2'], test_relation_inverse, train_factory.entity_to_id['-1']),
            tuple(test_leaked[0])
        )


def test_generate_compact_vectorized_lookup():
    """Test _generate_compact_vectorized_lookup."""
    max_id = 13
    ids = 2 * torch.randint(max_id, size=(2, 5))
    label_to_id = {
        f"e_{i}": i
        for i in range(2 * max_id)
    }
    new_label_to_id, mapping = _generate_compact_vectorized_lookup(
        ids=ids,
        label_to_id=label_to_id,
    )
    # test new label to ID
    # type
    assert isinstance(new_label_to_id, dict)
    # old labels
    assert set(new_label_to_id.keys()) == {f"e_{i}" for i in ids.unique().tolist()}
    # new, compact IDs
    assert set(new_label_to_id.values()) == set(range(len(ids.unique())))

    # test vectorized lookup
    # type
    assert torch.is_tensor(mapping)
    assert mapping.dtype == torch.long
    # shape
    assert mapping.shape == (ids.max() + 1,)
    # value range
    assert (mapping >= -1).all()
    # only occurring Ids get mapped to non-negative numbers
    assert set((mapping >= 0).nonzero().view(-1).tolist()) == set(ids.unique().tolist())
    # Ids are mapped to (0, ..., num_unique_ids-1)
    assert set(mapping[mapping >= 0].tolist()) == set(range(len(ids.unique())))
