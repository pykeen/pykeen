# -*- coding: utf-8 -*-

"""Unit tests for triples factories."""

import unittest

import numpy as np

from pykeen.datasets import Nations
from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory
from pykeen.triples.triples_factory import INVERSE_SUFFIX

triples = np.array(
    [
        ['peter', 'likes', 'chocolate_cake'],
        ['chocolate_cake', 'isA', 'dish'],
        ['susan', 'likes', 'pizza'],
        ['peter', 'likes', 'susan'],
    ],
    dtype=np.str,
)

instance_mapped_triples = np.array(
    [
        [0, 0],
        [2, 1],
        [4, 1],
    ],
)

instance_labels = np.array(
    [
        np.array([1]),
        np.array([0, 4]),
        np.array([3]),
    ],
)

numeric_triples = np.array(
    [
        ['peter', '/lit/hasAge', '30'],
        ['peter', '/lit/hasHeight', '185'],
        ['peter', '/lit/hasChildren', '2'],
        ['susan', '/lit/hasAge', '28'],
        ['susan', '/lit/hasHeight', '170'],
    ],
    dtype=np.str,
)


class TestTriplesFactory(unittest.TestCase):
    """Class for testing triples factories."""

    def test_correct_inverse_creation(self):
        """Test if the triples and the corresponding inverses are created and sorted correctly."""
        t = [
            ['e1', 'a.', 'e5'],
            ['e1', 'a', 'e2'],
        ]
        t = np.array(t, dtype=np.str)
        factory = TriplesFactory(triples=t, create_inverse_triples=True)
        reference_relation_to_id = {'a': 0, f'a{INVERSE_SUFFIX}': 1, 'a.': 2, f'a.{INVERSE_SUFFIX}': 3}
        self.assertEqual(reference_relation_to_id, factory.relation_to_id)

    def test_automatic_inverse_detection(self):
        """Test if the TriplesFactory detects that the triples contain inverses and creates correct ids."""
        t = [
            ['e3', f'a.{INVERSE_SUFFIX}', 'e10'],
            ['e1', 'a', 'e2'],
            ['e1', 'a.', 'e5'],
            ['e4', f'a{INVERSE_SUFFIX}', 'e5'],
        ]
        t = np.array(t, dtype=np.str)
        factory = TriplesFactory(triples=t, create_inverse_triples=False)
        reference_relation_to_id = {'a': 0, f'a{INVERSE_SUFFIX}': 1, 'a.': 2, f'a.{INVERSE_SUFFIX}': 3}
        self.assertEqual(reference_relation_to_id, factory.relation_to_id)
        self.assertTrue(factory.create_inverse_triples)

    def test_automatic_incomplete_inverse_detection(self):
        """Test if the TriplesFactory detects that the triples contain incomplete inverses and creates correct ids."""
        t = [
            ['e3', f'a.{INVERSE_SUFFIX}', 'e10'],
            ['e1', 'a', 'e2'],
            ['e1', 'a.', 'e5'],
        ]
        t = np.array(t, dtype=np.str)
        factory = TriplesFactory(triples=t, create_inverse_triples=False)
        reference_relation_to_id = {'a': 0, f'a{INVERSE_SUFFIX}': 1, 'a.': 2, f'a.{INVERSE_SUFFIX}': 3}
        self.assertEqual(reference_relation_to_id, factory.relation_to_id)
        self.assertTrue(factory.create_inverse_triples)

    def test_right_sorting(self):
        """Test if the triples and the corresponding inverses are sorted correctly."""
        t = [
            ['e1', 'a', 'e1'],
            ['e1', 'a.', 'e1'],
            ['e1', f'a.{INVERSE_SUFFIX}', 'e1'],
            ['e1', 'a.bc', 'e1'],
            ['e1', f'a.bc{INVERSE_SUFFIX}', 'e1'],
            ['e1', f'a{INVERSE_SUFFIX}', 'e1'],
            ['e1', 'abc', 'e1'],
            ['e1', f'abc{INVERSE_SUFFIX}', 'e1'],
        ]
        t = np.array(t, dtype=np.str)
        factory = TriplesFactory(triples=t, create_inverse_triples=False)
        reference_relation_to_id = {
            'a': 0,
            f'a{INVERSE_SUFFIX}': 1,
            'a.': 2,
            f'a.{INVERSE_SUFFIX}': 3,
            'a.bc': 4,
            f'a.bc{INVERSE_SUFFIX}': 5,
            'abc': 6,
            f'abc{INVERSE_SUFFIX}': 7,
        }
        self.assertEqual(reference_relation_to_id, factory.relation_to_id)


class TestSplit(unittest.TestCase):
    """Test splitting."""

    triples_factory: TriplesFactory

    def setUp(self) -> None:
        """Set up the tests."""
        self.triples_factory = Nations().training
        self.assertEqual(1592, self.triples_factory.num_triples)

    def test_split_naive(self):
        """Test splitting a factory in two with a given ratio."""
        ratio = 0.8
        train_triples_factory, test_triples_factory = self.triples_factory.split(ratio)
        expected_train_triples = int(self.triples_factory.num_triples * ratio)
        self.assertEqual(expected_train_triples, train_triples_factory.num_triples)
        self.assertEqual(self.triples_factory.num_triples - expected_train_triples, test_triples_factory.num_triples)

    def test_split_multi(self):
        """Test splitting a factory in three."""
        ratios = r0, r1 = 0.80, 0.10
        t0, t1, t2 = self.triples_factory.split(ratios)
        expected_0_triples = int(self.triples_factory.num_triples * r0)
        expected_1_triples = int(self.triples_factory.num_triples * r1)
        expected_2_triples = self.triples_factory.num_triples - expected_0_triples - expected_1_triples
        self.assertEqual(expected_0_triples, t0.num_triples)
        self.assertEqual(expected_1_triples, t1.num_triples)
        self.assertEqual(expected_2_triples, t2.num_triples)


class TestLiterals(unittest.TestCase):
    """Class for testing utils for processing numeric literals.tsv."""

    def test_create_lcwa_instances(self):
        """Test creating LCWA instances."""
        factory = TriplesNumericLiteralsFactory(triples=triples, numeric_triples=numeric_triples)
        instances = factory.create_lcwa_instances()

        id_peter = factory.entity_to_id['peter']
        id_age = instances.literals_to_id['/lit/hasAge']
        id_height = instances.literals_to_id['/lit/hasHeight']
        id_num_children = instances.literals_to_id['/lit/hasChildren']

        self.assertEqual(instances.numeric_literals[id_peter, id_age], 30)
        self.assertEqual(instances.numeric_literals[id_peter, id_height], 185)
        self.assertEqual(instances.numeric_literals[id_peter, id_num_children], 2)

        id_susan = factory.entity_to_id['susan']
        id_age = instances.literals_to_id['/lit/hasAge']
        id_height = instances.literals_to_id['/lit/hasHeight']
        id_num_children = instances.literals_to_id['/lit/hasChildren']

        self.assertEqual(instances.numeric_literals[id_susan, id_age], 28)
        self.assertEqual(instances.numeric_literals[id_susan, id_height], 170)
        self.assertEqual(instances.numeric_literals[id_susan, id_num_children], 0)

        id_chocolate_cake = factory.entity_to_id['chocolate_cake']
        id_age = instances.literals_to_id['/lit/hasAge']
        id_height = instances.literals_to_id['/lit/hasHeight']
        id_num_children = instances.literals_to_id['/lit/hasChildren']

        self.assertEqual(instances.numeric_literals[id_chocolate_cake, id_age], 0)
        self.assertEqual(instances.numeric_literals[id_chocolate_cake, id_height], 0)
        self.assertEqual(instances.numeric_literals[id_chocolate_cake, id_num_children], 0)

        # Check if multilabels are working correctly
        self.assertTrue((instance_mapped_triples == instances.mapped_triples.cpu().detach().numpy()).all())
        self.assertTrue(all(all(instance_labels[i] == instances.labels[i]) for i in range(len(instance_labels))))

    def test_triples(self):
        """Test properties of the triples factory."""
        triples_factory = TriplesFactory(triples=triples)
        self.assertEqual(set(range(triples_factory.num_entities)), set(triples_factory.entity_to_id.values()))
        self.assertEqual(set(range(triples_factory.num_relations)), set(triples_factory.relation_to_id.values()))
        self.assertTrue((triples_factory.mapped_triples == triples_factory.map_triples_to_id(triples)).all())

    def test_inverse_triples(self):
        """Test that the right number of entities and triples exist after inverting them."""
        triples_factory = TriplesFactory(triples=triples, create_inverse_triples=True)
        self.assertEqual(0, triples_factory.num_relations % 2)
        self.assertEqual(
            set(range(triples_factory.num_entities)),
            set(triples_factory.entity_to_id.values()),
            msg='wrong number entities',
        )
        self.assertEqual(
            set(range(triples_factory.num_relations)),
            set(triples_factory.relation_to_id.values()),
            msg='wrong number relations',
        )

        relations = set(triples[:, 1])
        entities = set(triples[:, 0]).union(triples[:, 2])
        self.assertEqual(len(entities), triples_factory.num_entities, msg='wrong number entities')
        self.assertEqual(2, len(relations), msg='Wrong number of relations in set')
        self.assertEqual(
            2 * len(relations),
            triples_factory.num_relations,
            msg='Wrong number of relations in factory',
        )

        self.assertIn(f'likes{INVERSE_SUFFIX}', triples_factory.relation_to_id)
