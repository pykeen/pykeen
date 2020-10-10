# -*- coding: utf-8 -*-

"""Unit tests for triples factories."""

import unittest

import numpy as np
import torch

from pykeen.datasets import Nations
from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory
from pykeen.triples.triples_factory import (
    INVERSE_SUFFIX, TRIPLES_DF_COLUMNS, _tf_cleanup_all, _tf_cleanup_deterministic, _tf_cleanup_randomized,
)

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
    dtype=object,
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

    def setUp(self) -> None:
        """Instantiate test instance."""
        self.factory = Nations().training

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

    def test_id_to_label(self):
        """Test ID-to-label conversion."""
        for label_to_id, id_to_label in [
            (self.factory.entity_to_id, self.factory.entity_id_to_label),
            (self.factory.relation_to_id, self.factory.relation_id_to_label),
        ]:
            for k in label_to_id.keys():
                assert id_to_label[label_to_id[k]] == k
            for k in id_to_label.keys():
                assert label_to_id[id_to_label[k]] == k

    def test_tensor_to_df(self):
        """Test tensor_to_df()."""
        # check correct translation
        labeled_triples = set(tuple(row) for row in self.factory.triples.tolist())
        tensor = self.factory.mapped_triples
        scores = torch.rand(tensor.shape[0])
        df = self.factory.tensor_to_df(tensor=tensor, scores=scores)
        re_labeled_triples = set(
            tuple(row)
            for row in df[['head_label', 'relation_label', 'tail_label']].values.tolist()
        )
        assert labeled_triples == re_labeled_triples

        # check column order
        assert tuple(df.columns) == TRIPLES_DF_COLUMNS + ('scores',)

    def test_new_with_restriction(self):
        """Test new_with_restriction()."""
        example_relation_restriction = {
            'economicaid',
            'dependent',
        }
        example_entity_restriction = {
            'brazil',
            'burma',
            'china',
        }
        for inverse_triples in (True, False):
            original_triples_factory = Nations(
                create_inverse_triples=inverse_triples,
            ).training
            for entity_restriction in (None, example_entity_restriction):
                for relation_restriction in (None, example_relation_restriction):
                    # apply restriction
                    restricted_triples_factory = original_triples_factory.new_with_restriction(
                        entities=entity_restriction,
                        relations=relation_restriction,
                    )
                    # check that the triples factory is returned as is, if and only if no restriction is to apply
                    no_restriction_to_apply = (entity_restriction is None and relation_restriction is None)
                    equal_factory_object = (id(restricted_triples_factory) == id(original_triples_factory))
                    assert no_restriction_to_apply == equal_factory_object

                    # check that inverse_triples is correctly carried over
                    assert (
                        original_triples_factory.create_inverse_triples
                        == restricted_triples_factory.create_inverse_triples
                    )

                    # verify that the label-to-ID mapping has not been changed
                    assert original_triples_factory.entity_to_id == restricted_triples_factory.entity_to_id
                    assert original_triples_factory.relation_to_id == restricted_triples_factory.relation_to_id

                    # verify that triples have been filtered
                    if entity_restriction is not None:
                        present_relations = set(restricted_triples_factory.triples[:, 0]).union(
                            restricted_triples_factory.triples[:, 2])
                        assert set(entity_restriction).issuperset(present_relations)

                    if relation_restriction is not None:
                        present_relations = set(restricted_triples_factory.triples[:, 1])
                        exp_relations = set(relation_restriction)
                        if original_triples_factory.create_inverse_triples:
                            exp_relations = exp_relations.union(map(original_triples_factory.relation_to_inverse.get,
                                                                    exp_relations))
                        assert exp_relations.issuperset(present_relations)


class TestSplit(unittest.TestCase):
    """Test splitting."""

    triples_factory: TriplesFactory

    def setUp(self) -> None:
        """Set up the tests."""
        self.triples_factory = Nations().training
        self.assertEqual(1592, self.triples_factory.num_triples)

    def _test_invariants(self, training_triples_factory: TriplesFactory, *other_factories: TriplesFactory) -> None:
        """Test invariants for result of triples factory splitting."""
        # verify that all entities and relations are present in the training factory
        assert training_triples_factory.num_entities == self.triples_factory.num_entities
        assert training_triples_factory.num_relations == self.triples_factory.num_relations

        all_factories = (training_triples_factory,) + other_factories

        # verify that no triple got lost
        self.assertEqual(sum(t.num_triples for t in all_factories), self.triples_factory.num_triples)

        # verify that the label-to-id mappings match
        self.assertSetEqual({
            id(factory.entity_to_id)
            for factory in all_factories
        }, {
            id(self.triples_factory.entity_to_id),
        })
        self.assertSetEqual({
            id(factory.relation_to_id)
            for factory in all_factories
        }, {
            id(self.triples_factory.relation_to_id),
        })

    def test_split_naive(self):
        """Test splitting a factory in two with a given ratio."""
        ratio = 0.8
        train_triples_factory, test_triples_factory = self.triples_factory.split(ratio)
        self._test_invariants(train_triples_factory, test_triples_factory)

    def test_split_multi(self):
        """Test splitting a factory in three."""
        ratios = 0.80, 0.10
        t0, t1, t2 = self.triples_factory.split(ratios)
        self._test_invariants(t0, t1, t2)

    def test_cleanup_deterministic(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = np.array([
            [1, 1000, 2],
            [1, 1000, 3],
            [1, 1001, 3],
        ])
        testing = np.array([
            [2, 1001, 3],
            [1, 1002, 4],
        ])
        expected_training = [
            [1, 1000, 2],
            [1, 1000, 3],
            [1, 1001, 3],
            [1, 1002, 4],
        ]
        expected_testing = [
            [2, 1001, 3],
        ]

        new_training, new_testing = _tf_cleanup_deterministic(training, testing)
        self.assertEqual(expected_training, new_training.tolist())
        self.assertEqual(expected_testing, new_testing.tolist())

        new_testing, new_testing = _tf_cleanup_all([training, testing])
        self.assertEqual(expected_training, new_training.tolist())
        self.assertEqual(expected_testing, new_testing.tolist())

    def test_cleanup_randomized(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = np.array([
            [1, 1000, 2],
            [1, 1000, 3],
        ])
        testing = np.array([
            [2, 1000, 3],
            [1, 1000, 4],
            [2, 1000, 4],
            [1, 1001, 3],
        ])
        expected_training_1 = {
            (1, 1000, 2),
            (1, 1000, 3),
            (1, 1000, 4),
            (1, 1001, 3),
        }
        expected_testing_1 = {
            (2, 1000, 3),
            (2, 1000, 4),
        }

        expected_training_2 = {
            (1, 1000, 2),
            (1, 1000, 3),
            (2, 1000, 4),
            (1, 1001, 3),
        }
        expected_testing_2 = {
            (2, 1000, 3),
            (1, 1000, 4),
        }

        new_training, new_testing = [
            set(tuple(row) for row in arr.tolist())
            for arr in _tf_cleanup_randomized(training, testing)
        ]

        if expected_training_1 == new_training:
            self.assertEqual(expected_testing_1, new_testing)
        elif expected_training_2 == new_training:
            self.assertEqual(expected_testing_2, new_testing)
        else:
            self.fail('training was not correct')


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
