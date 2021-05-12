# -*- coding: utf-8 -*-

"""Unit tests for triples factories."""

import itertools as itt
import os
import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from pykeen.triples import LCWAInstances, TriplesFactory, TriplesNumericLiteralsFactory
from pykeen.triples.generation import generate_triples
from pykeen.triples.splitting import (
    SPLIT_METHODS, _get_cover_deterministic, _tf_cleanup_all, _tf_cleanup_deterministic,
    _tf_cleanup_randomized,
    get_absolute_split_sizes, normalize_ratios,
)
from pykeen.triples.triples_factory import INVERSE_SUFFIX, TRIPLES_DF_COLUMNS, _map_triples_elements_to_ids
from pykeen.triples.utils import get_entities, get_relations, load_triples
from tests.constants import RESOURCES

triples = np.array(
    [
        ['peter', 'likes', 'chocolate_cake'],
        ['chocolate_cake', 'isA', 'dish'],
        ['susan', 'likes', 'pizza'],
        ['peter', 'likes', 'susan'],
    ],
    dtype=str,
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
    dtype=str,
)


class TestTriplesFactory(unittest.TestCase):
    """Class for testing triples factories."""

    def setUp(self) -> None:
        """Instantiate test instance."""
        self.factory = Nations().training

    def test_correct_inverse_creation(self):
        """Test if the triples and the corresponding inverses are created."""
        t = [
            ['e1', 'a.', 'e5'],
            ['e1', 'a', 'e2'],
        ]
        t = np.array(t, dtype=str)
        factory = TriplesFactory.from_labeled_triples(triples=t, create_inverse_triples=True)
        instances = factory.create_slcwa_instances()
        assert len(instances) == 4

    def test_automatic_incomplete_inverse_detection(self):
        """Test detecting that the triples contain inverses, warns about them, and filters them out."""
        # comment(mberr): from my pov this behaviour is faulty: the triples factory is expected to say it contains
        # inverse relations, although the triples contained in it are not the same we would have when removing the
        # first triple, and passing create_inverse_triples=True.
        t = [
            ['e3', f'a.{INVERSE_SUFFIX}', 'e10'],
            ['e1', 'a', 'e2'],
            ['e1', 'a.', 'e5'],
        ]
        t = np.array(t, dtype=str)
        for create_inverse_triples in (False, True):
            with patch("pykeen.triples.triples_factory.logger.warning") as warning:
                factory = TriplesFactory.from_labeled_triples(triples=t, create_inverse_triples=create_inverse_triples)
                # check for warning
                warning.assert_called()
                # check for filtered triples
                assert factory.num_triples == 2
                # check for correct inverse triples flag
                assert factory.create_inverse_triples == create_inverse_triples

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
                        present_entities = set(restricted_triples_factory.triples[:, 0]).union(
                            restricted_triples_factory.triples[:, 2])
                        assert set(entity_restriction).issuperset(present_entities)

                    if relation_restriction is not None:
                        present_relations = set(restricted_triples_factory.triples[:, 1])
                        exp_relations = set(relation_restriction)
                        assert exp_relations.issuperset(present_relations)

    def test_create_lcwa_instances(self):
        """Test create_lcwa_instances."""
        factory = Nations().training
        instances = factory.create_lcwa_instances()
        assert isinstance(instances, LCWAInstances)

        # check compressed triples
        # reconstruct triples from compressed form
        reconstructed_triples = set()
        for hr, row_id in zip(instances.pairs, range(instances.compressed.shape[0])):
            h, r = hr.tolist()
            _, tails = instances.compressed[row_id].nonzero()
            reconstructed_triples.update(
                (h, r, t)
                for t in tails.tolist()
            )
        original_triples = {
            tuple(hrt)
            for hrt in factory.mapped_triples.tolist()
        }
        assert original_triples == reconstructed_triples

        # check data loader
        for batch in torch.utils.data.DataLoader(instances, batch_size=2):
            assert len(batch) == 2
            assert all(torch.is_tensor(x) for x in batch)
            x, y = batch
            batch_size = x.shape[0]
            assert x.shape == (batch_size, 2)
            assert x.dtype == torch.long
            assert y.shape == (batch_size, factory.num_entities)
            assert y.dtype == torch.get_default_dtype()

    def test_split_inverse_triples(self):
        """Test whether inverse triples are only created in the training factory."""
        # set create inverse triple to true
        self.factory.create_inverse_triples = True
        # split factory
        train, *others = self.factory.split()
        # check that in *training* inverse triple are to be created
        assert train.create_inverse_triples
        # check that in all other splits no inverse triples are to be created
        assert not any(f.create_inverse_triples for f in others)


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
        self.assertEqual(training_triples_factory.num_entities, self.triples_factory.num_entities)
        self.assertEqual(training_triples_factory.num_relations, self.triples_factory.num_relations)

        all_factories = (training_triples_factory, *other_factories)

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

    def test_split(self):
        """Test splitting a factory."""
        cases = [
            (2, 0.8),
            (2, [0.8]),
            (3, [0.80, 0.10]),
            (3, [0.80, 0.10, 0.10]),
        ]
        for method, (n, ratios), in itt.product(SPLIT_METHODS, cases):
            with self.subTest(method=method, ratios=ratios):
                factories = self.triples_factory.split(ratios, method=method)
                self.assertEqual(n, len(factories))
                self._test_invariants(*factories)

    def test_cleanup_deterministic(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = torch.as_tensor(data=[
            [1, 1000, 2],
            [1, 1000, 3],
            [1, 1001, 3],
        ], dtype=torch.long)
        testing = torch.as_tensor(data=[
            [2, 1001, 3],
            [1, 1002, 4],
        ], dtype=torch.long)
        expected_training = torch.as_tensor(data=[
            [1, 1000, 2],
            [1, 1000, 3],
            [1, 1001, 3],
            [1, 1002, 4],
        ], dtype=torch.long)
        expected_testing = torch.as_tensor(data=[
            [2, 1001, 3],
        ], dtype=torch.long)

        new_training, new_testing = _tf_cleanup_deterministic(training, testing)
        assert (expected_training == new_training).all()
        assert (expected_testing == new_testing).all()

        new_testing, new_testing = _tf_cleanup_all([training, testing])
        assert (expected_training == new_training).all()
        assert (expected_testing == new_testing).all()

    def test_cleanup_randomized(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = torch.as_tensor(data=[
            [1, 1000, 2],
            [1, 1000, 3],
        ], dtype=torch.long)
        testing = torch.as_tensor(data=[
            [2, 1000, 3],
            [1, 1000, 4],
            [2, 1000, 4],
            [1, 1001, 3],
        ], dtype=torch.long)
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

    def test_get_cover_deterministic(self):
        """Test _get_cover_deterministic."""
        generated_triples = generate_triples()
        cover = _get_cover_deterministic(triples=generated_triples)

        # check type
        assert torch.is_tensor(cover)
        assert cover.dtype == torch.bool
        # check format
        assert cover.shape == (generated_triples.shape[0],)

        # check coverage
        self.assertEqual(
            get_entities(generated_triples),
            get_entities(generated_triples[cover]),
            msg='entity coverage is not full',
        )
        self.assertEqual(
            get_relations(generated_triples),
            get_relations(generated_triples[cover]),
            msg='relation coverage is not full',
        )


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
        self.assertTrue((instance_mapped_triples == instances.pairs).all())
        for i, exp in enumerate(instance_labels):
            self.assertTrue((exp == instances.compressed[i].nonzero()[-1]).all())

    def test_triples(self):
        """Test properties of the triples factory."""
        triples_factory = TriplesFactory.from_labeled_triples(triples=triples)
        self.assertEqual(set(range(triples_factory.num_entities)), set(triples_factory.entity_to_id.values()))
        self.assertEqual(set(range(triples_factory.num_relations)), set(triples_factory.relation_to_id.values()))
        assert (_map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=triples_factory.entity_to_id,
            relation_to_id=triples_factory.relation_to_id,
        ) == triples_factory.mapped_triples).all()

    def test_inverse_triples(self):
        """Test that the right number of entities and triples exist after inverting them."""
        triples_factory = TriplesFactory.from_labeled_triples(triples=triples, create_inverse_triples=True)
        self.assertEqual(4, triples_factory.num_relations)
        self.assertEqual(
            set(range(triples_factory.num_entities)),
            set(triples_factory.entity_to_id.values()),
            msg='wrong number entities',
        )
        self.assertEqual(
            set(range(triples_factory.real_num_relations)),
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

    def test_metadata(self):
        """Test metadata passing for triples factories."""
        t = Nations().training
        self.assertEqual(NATIONS_TRAIN_PATH, t.metadata['path'])
        self.assertEqual(
            (
                f'TriplesFactory(num_entities=14, num_relations=55, num_triples=1592,'
                f' inverse_triples=False, path="{NATIONS_TRAIN_PATH}")'
            ),
            repr(t),
        )

        entities = ['poland', 'ussr']
        x = t.new_with_restriction(entities=entities)
        entities_ids = t.entities_to_ids(entities=entities)
        self.assertEqual(NATIONS_TRAIN_PATH, x.metadata['path'])
        self.assertEqual(
            (
                f'TriplesFactory(num_entities=14, num_relations=55, num_triples=37,'
                f' inverse_triples=False, entity_restriction={repr(entities_ids)}, path="{NATIONS_TRAIN_PATH}")'
            ),
            repr(x),
        )

        relations = ['negativebehavior']
        v = t.new_with_restriction(relations=relations)
        relations_ids = t.relations_to_ids(relations=relations)
        self.assertEqual(NATIONS_TRAIN_PATH, x.metadata['path'])
        self.assertEqual(
            (
                f'TriplesFactory(num_entities=14, num_relations=55, num_triples=29,'
                f' inverse_triples=False, path="{NATIONS_TRAIN_PATH}", relation_restriction={repr(relations_ids)})'
            ),
            repr(v),
        )

        w = t.clone_and_exchange_triples(t.triples[0:5], keep_metadata=False)
        self.assertIsInstance(w, TriplesFactory)
        self.assertNotIn('path', w.metadata)
        self.assertEqual(
            'TriplesFactory(num_entities=14, num_relations=55, num_triples=5, inverse_triples=False)',
            repr(w),
        )

        y, z = t.split()
        self.assertEqual(NATIONS_TRAIN_PATH, y.metadata['path'])
        self.assertEqual(NATIONS_TRAIN_PATH, z.metadata['path'])


class TestUtils(unittest.TestCase):
    """Test triples utilities."""

    def test_load_triples_remapped(self):
        """Test loading a triples file where the columns must be remapped."""
        path = os.path.join(RESOURCES, 'test_remap.tsv')

        with self.assertRaises(ValueError):
            load_triples(path, column_remapping=[1, 2])

        _triples = load_triples(path, column_remapping=[0, 2, 1])
        self.assertEqual(
            [
                ['a', 'r1', 'b'],
                ['b', 'r2', 'c'],
            ],
            _triples.tolist(),
        )


def test_get_absolute_split_sizes():
    """Test get_absolute_split_sizes."""
    for num_splits, n_total in zip(
        (2, 3, 4),
        (100, 200, 10412),
    ):
        # generate random ratios
        ratios = np.random.uniform(size=(num_splits,))
        ratios = ratios / ratios.sum()
        sizes = get_absolute_split_sizes(n_total=n_total, ratios=ratios)
        # check size
        assert len(sizes) == len(ratios)

        # check value range
        assert all(0 <= size <= n_total for size in sizes)

        # check total split
        assert sum(sizes) == n_total

        # check consistency with ratios
        rel_size = np.asarray(sizes) / n_total
        # the number of decimal digits equivalent to 1 / n_total
        decimal = np.floor(np.log10(n_total))
        np.testing.assert_almost_equal(rel_size, ratios, decimal=decimal)


def test_normalize_ratios():
    """Test normalize_ratios."""
    for ratios, exp_output in (
        (0.5, (0.5, 0.5)),
        ((0.3, 0.2, 0.4), (0.3, 0.2, 0.4, 0.1)),
        ((0.3, 0.3, 0.4), (0.3, 0.3, 0.4)),
    ):
        output = normalize_ratios(ratios=ratios)
        # check type
        assert isinstance(output, tuple)
        assert all(isinstance(ratio, float) for ratio in output)
        # check values
        assert len(output) >= 2
        assert all(0 <= ratio <= 1 for ratio in output)
        output_np = np.asarray(output)
        np.testing.assert_almost_equal(output_np.sum(), np.ones(1))
        # compare against expected
        np.testing.assert_almost_equal(output_np, np.asarray(exp_output))


def test_normalize_invalid_ratio():
    """Test invalid ratios."""
    cases = [
        1.1,
        [1.1],
        [0.8, 0.3],
        [0.8, 0.1, 0.2],
    ]
    for ratios in cases:
        with pytest.raises(ValueError):
            _ = normalize_ratios(ratios=ratios)
