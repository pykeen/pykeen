# -*- coding: utf-8 -*-

"""Tests for leakage analysis."""

import itertools as itt
import unittest

import numpy
import numpy as np
import scipy.sparse
import torch

from pykeen.datasets import Nations
from pykeen.triples import TriplesFactory
from pykeen.triples.leakage import (
    Sealant, _generate_compact_vectorized_lookup, _translate_triples, get_candidate_pairs, jaccard_similarity_scipy,
    mapped_triples_to_sparse_matrices, triples_factory_to_sparse_matrices,
)


class TestLeakage(unittest.TestCase):
    """Tests for identifying inverse relationships and leakage."""

    @unittest.skip('need to reinvestigate leakage pipeline')
    def test_find_leak_assymetric(self):
        """Test finding test leakages with an asymmetric metric."""
        n = 100
        min_frequency = 0.97
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

        expected_forwards_frequency = n / (n + len(forwards_extras))
        expected_inverse_frequency = n / (n + len(inverse_extras))
        # expected_frequency = n / (n + len(forwards_extras) + len(inverse_extras))
        # self.assertLessEqual(min_frequency, expected_frequency)

        self.assertGreater(len(forwards_extras), len(inverse_extras))
        self.assertLess(
            expected_forwards_frequency, expected_inverse_frequency,
            msg='Forwards frequency should be higher than inverse frequency',
        )

        sealant = Sealant(train_factory, symmetric=False, minimum_frequency=min_frequency)
        test_relation_id, test_relation_inverse_id = [
            train_factory.relation_to_id[r] for r in (test_relation, test_relation_inverse)
        ]
        self.assertNotEqual(
            0, len(sealant.candidate_inverse_relations),
            msg=f'did not find any candidate inverse relations at frequency>={min_frequency}',
        )
        self.assertEqual(
            {
                (test_relation_id, test_relation_inverse_id): expected_forwards_frequency,
                (test_relation_inverse_id, test_relation_id): expected_inverse_frequency,
            },
            dict(sealant.candidate_inverse_relations),
        )

        self.assertIn(test_relation_id, sealant.inverses)
        self.assertEqual(test_relation_inverse_id, sealant.inverses[test_relation])
        self.assertIn(test_relation_inverse_id, sealant.inverses)
        self.assertEqual(test_relation, sealant.inverses[test_relation_inverse_id])

        self.assertIn(
            test_relation_inverse_id,
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
            tuple(test_leaked[0]),
        )

    def test_generate_compact_vectorized_lookup(self):
        """Test :func:`_generate_compact_vectorized_lookup`."""
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

    def test_translate_triples(self):
        """Test :func:`_translate_triples`."""
        max_e_id = 13
        max_r_id = 7
        num_triples = 31
        triples = torch.stack([
            2 * torch.randint(max_id, size=(num_triples,))
            for max_id in (max_e_id, max_r_id, max_e_id)
        ], dim=-1)
        entity_translation = torch.full(size=(2 * max_e_id,), fill_value=-1, dtype=torch.long)
        entity_translation[::2] = torch.arange(max_e_id)
        relation_translation = torch.full(size=(2 * max_r_id,), fill_value=-1, dtype=torch.long)
        relation_translation[::2] = torch.arange(max_r_id)
        new_triples = _translate_triples(
            triples=triples,
            entity_translation=entity_translation,
            relation_translation=relation_translation,
        )
        # check type
        assert torch.is_tensor(new_triples)
        assert new_triples.dtype == torch.long
        # check shape
        assert new_triples.shape == (num_triples, 3)
        # check content
        assert (new_triples >= 0).all()
        assert (new_triples[:, [0, 2]] < max_e_id).all()
        assert (new_triples[:, 1] < max_r_id).all()

    def test_relations_to_sparse_matrices(self):
        """Test :func:`triples_factory_to_sparse_matrices`."""
        triples_factory = Nations().training
        rel, inv = triples_factory_to_sparse_matrices(triples_factory)
        for m in (rel, inv):
            # check type
            assert isinstance(m, scipy.sparse.spmatrix)
            assert m.dtype == numpy.int32
            # check shape
            assert m.shape[0] == triples_factory.num_relations
            # check 1-hot
            assert m.max() == 1

    def test_jaccard_similarity_scipy(self):
        """Test :func:`jaccard_similarity_scipy`."""
        triples_factory = Nations().training
        rel, inv = triples_factory_to_sparse_matrices(triples_factory)
        sim = jaccard_similarity_scipy(a=rel, b=rel)
        # check type
        assert isinstance(sim, numpy.ndarray)
        assert sim.dtype == numpy.float64
        # check shape
        assert sim.shape == (triples_factory.num_relations, triples_factory.num_relations)
        # check value range
        assert (sim >= 0).all()
        assert (sim <= 1).all()
        # check self-similarity = 1
        numpy.testing.assert_allclose(numpy.diag(sim), 1.0)

    def test_candidate_pairs(self):
        """Test :func:`get_candidate_pairs`."""
        num_entities = 11
        num_pairs = 100
        h, t = torch.randint(num_entities, size=(2, num_pairs))
        r = torch.zeros_like(h)

        # base relation
        base = torch.stack([h, r, t], dim=-1)

        # exact duplicate
        dup100 = torch.stack([h, r + 1, t], dim=-1)

        # 99% duplicate
        dup099 = torch.stack([h, r + 2, t], dim=-1)
        dup099[99:, [0, 2]] += 1

        # 50% duplicate
        dup050 = torch.stack([h, r + 3, t], dim=-1)
        dup050[50:, [0, 2]] += 1

        # exact inverse
        inv100 = torch.stack([t, r + 4, h], dim=-1)

        # 99% inverse
        inv099 = torch.stack([t, r + 5, h], dim=-1)
        inv099[99:, [0, 2]] += 1

        triples = torch.cat([base, dup100, dup099, dup050, inv100, inv099], dim=0)

        rel, inv = mapped_triples_to_sparse_matrices(
            triples,
            num_relations=6,
        )
        assert rel.max() == 1 and inv.max() == 1
        candidate_pairs = get_candidate_pairs(a=rel, threshold=0.97)
        expected_candidate_pairs = {
            (0, 1), (0, 2),
            (1, 0), (1, 2),
            (2, 0), (2, 1),
            (4, 5),
            (5, 4),
        }
        self.assertEqual(expected_candidate_pairs, candidate_pairs)

        candidate_pairs = get_candidate_pairs(a=rel, b=inv, threshold=0.97)
        expected_candidate_pairs = {
            (0, 4), (0, 5),
            (1, 4), (1, 5),
            (2, 4), (2, 5),
            (4, 0), (4, 1), (4, 2),
            (5, 0), (5, 1), (5, 2),
        }
        self.assertEqual(expected_candidate_pairs, candidate_pairs)
