# -*- coding: utf-8 -*-

"""Unit tests for triples factories."""

import itertools as itt
import os
import tempfile
import unittest
from pathlib import Path
from typing import Collection, Optional
from unittest.mock import patch

import numpy as np
import torch

from pykeen.datasets import Hetionet, Nations, SingleTabbedDataset
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from pykeen.triples import CoreTriplesFactory, LCWAInstances, TriplesFactory, TriplesNumericLiteralsFactory
from pykeen.triples.splitting import splitter_resolver
from pykeen.triples.triples_factory import INVERSE_SUFFIX, _map_triples_elements_to_ids
from pykeen.triples.utils import TRIPLES_DF_COLUMNS, load_triples
from tests.constants import RESOURCES

triples = np.array(
    [
        ["peter", "likes", "chocolate_cake"],
        ["chocolate_cake", "isA", "dish"],
        ["susan", "likes", "pizza"],
        ["peter", "likes", "susan"],
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
        ["peter", "/lit/hasAge", "30"],
        ["peter", "/lit/hasHeight", "185"],
        ["peter", "/lit/hasChildren", "2"],
        ["susan", "/lit/hasAge", "28"],
        ["susan", "/lit/hasHeight", "170"],
    ],
    dtype=str,
)

# See https://github.com/pykeen/pykeen/pull/883
triples_with_nans = [
    ["netherlands", "militaryalliance", "uk"],
    ["egypt", "intergovorgs3", "usa"],
    ["jordan", "relbooktranslations", "nan"],
]


class TestTriplesFactory(unittest.TestCase):
    """Class for testing triples factories."""

    def setUp(self) -> None:
        """Instantiate test instance."""
        self.factory = Nations().training

    def test_correct_inverse_creation(self):
        """Test if the triples and the corresponding inverses are created."""
        t = [
            ["e1", "a.", "e5"],
            ["e1", "a", "e2"],
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
            ["e3", f"a.{INVERSE_SUFFIX}", "e10"],
            ["e1", "a", "e2"],
            ["e1", "a.", "e5"],
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
            tuple(row) for row in df[["head_label", "relation_label", "tail_label"]].values.tolist()
        )
        assert labeled_triples == re_labeled_triples

        # check column order
        assert tuple(df.columns) == TRIPLES_DF_COLUMNS + ("scores",)

    def _test_restriction(
        self,
        original_triples_factory: TriplesFactory,
        entity_restriction: Optional[Collection[str]],
        invert_entity_selection: bool,
        relation_restriction: Optional[Collection[str]],
        invert_relation_selection: bool,
    ):
        """Run the actual test for new_with_restriction."""
        # apply restriction
        restricted_triples_factory = original_triples_factory.new_with_restriction(
            entities=entity_restriction,
            relations=relation_restriction,
            invert_entity_selection=invert_entity_selection,
            invert_relation_selection=invert_relation_selection,
        )

        # check that the triples factory is returned as is, if and only if no restriction is to apply
        no_restriction_to_apply = entity_restriction is None and relation_restriction is None
        equal_factory_object = id(restricted_triples_factory) == id(original_triples_factory)
        assert no_restriction_to_apply == equal_factory_object

        # check that inverse_triples is correctly carried over
        assert original_triples_factory.create_inverse_triples == restricted_triples_factory.create_inverse_triples

        # verify that the label-to-ID mapping has not been changed
        assert original_triples_factory.entity_to_id == restricted_triples_factory.entity_to_id
        assert original_triples_factory.relation_to_id == restricted_triples_factory.relation_to_id

        # verify that triples have been filtered
        if entity_restriction is not None:
            present_entities = set(restricted_triples_factory.triples[:, 0]).union(
                restricted_triples_factory.triples[:, 2]
            )
            expected_entities = (
                set(original_triples_factory.entity_id_to_label.values()).difference(entity_restriction)
                if invert_entity_selection
                else entity_restriction
            )
            assert expected_entities.issuperset(present_entities)

        if relation_restriction is not None:
            present_relations = set(restricted_triples_factory.triples[:, 1])
            expected_relations = (
                set(original_triples_factory.relation_id_to_label.values())
                if invert_relation_selection
                else set(relation_restriction)
            )
            assert expected_relations.issuperset(present_relations)

    def test_new_with_restriction(self):
        """Test new_with_restriction()."""
        relation_restrictions = {
            "economicaid",
            "dependent",
        }
        entity_restrictions = {
            "brazil",
            "burma",
            "china",
        }
        for inverse_triples in (True, False):
            original_triples_factory = Nations(
                create_inverse_triples=inverse_triples,
            ).training
            # Test different combinations of restrictions
            for (
                (entity_restriction, invert_entity_selection),
                (relation_restriction, invert_relation_selection),
            ) in itt.product(
                ((None, None), (entity_restrictions, False), (entity_restrictions, True)),
                ((None, None), (relation_restrictions, False), (relation_restrictions, True)),
            ):
                with self.subTest(
                    entity_restriction=entity_restriction,
                    invert_entity_selection=invert_entity_selection,
                    relation_restriction=relation_restriction,
                    invert_relation_selection=invert_relation_selection,
                ):
                    self._test_restriction(
                        original_triples_factory=original_triples_factory,
                        entity_restriction=entity_restriction,
                        invert_entity_selection=invert_entity_selection,
                        relation_restriction=relation_restriction,
                        invert_relation_selection=invert_relation_selection,
                    )

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
            reconstructed_triples.update((h, r, t) for t in tails.tolist())
        original_triples = {tuple(hrt) for hrt in factory.mapped_triples.tolist()}
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
        self.dataset = Nations()
        self.triples_factory = self.dataset.training
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
        self.assertSetEqual(
            {id(factory.entity_to_id) for factory in all_factories},
            {
                id(self.triples_factory.entity_to_id),
            },
        )
        self.assertSetEqual(
            {id(factory.relation_to_id) for factory in all_factories},
            {
                id(self.triples_factory.relation_to_id),
            },
        )

    def test_split_tf(self):
        """Test splitting a factory."""
        cases = [
            (2, 0.8),
            (2, [0.8]),
            (3, [0.80, 0.10]),
            (3, [0.80, 0.10, 0.10]),
        ]
        for (
            method,
            (n, ratios),
        ) in itt.product(splitter_resolver.options, cases):
            with self.subTest(method=method, ratios=ratios):
                factories_1 = self.triples_factory.split(ratios, method=method, random_state=0)
                self.assertEqual(n, len(factories_1))
                self._test_invariants(*factories_1)

                factories_2 = self.triples_factory.split(ratios, method=method, random_state=0)
                self.assertEqual(n, len(factories_2))
                self._test_invariants(*factories_2)

                self._compare_factories(factories_1, factories_2)

    def test_load_model(self):
        """Test splitting a tabbed dataset."""

        class MockSingleTabbedDataset(SingleTabbedDataset):
            def __init__(self, random_state=0, **kwargs):
                super().__init__(url=NATIONS_TRAIN_PATH.as_uri(), random_state=random_state, **kwargs)

        dataset_classes = [MockSingleTabbedDataset]
        if Hetionet(eager=False)._get_path().is_file():
            dataset_classes.append(Hetionet)

        for cls in dataset_classes:
            with self.subTest(name=cls.__name__):
                self._test_random_dataset(cls)

    def _test_random_dataset(self, cls) -> None:
        ds1 = cls(random_state=0)
        ds2 = cls(random_state=0)
        self._compare_factories(
            (ds1.training, ds1.testing, ds1.validation),
            (ds2.training, ds2.testing, ds2.validation),
            msg=f"Failed on {ds1.__class__.__name__}",
        )

    def _compare_factories(self, factories_1, factories_2, msg=None) -> None:
        for factory_1, factory_2 in zip(factories_1, factories_2):
            triples_1 = factory_1.mapped_triples.detach().cpu().numpy()
            triples_2 = factory_2.mapped_triples.detach().cpu().numpy()
            self.assertTrue((triples_1 == triples_2).all(), msg=msg)


class TestLiterals(unittest.TestCase):
    """Class for testing utils for processing numeric literals.tsv."""

    def test_triples(self):
        """Test properties of the triples factory."""
        triples_factory = TriplesFactory.from_labeled_triples(triples=triples)
        self.assertEqual(set(range(triples_factory.num_entities)), set(triples_factory.entity_to_id.values()))
        self.assertEqual(set(range(triples_factory.num_relations)), set(triples_factory.relation_to_id.values()))
        assert (
            _map_triples_elements_to_ids(
                triples=triples,
                entity_to_id=triples_factory.entity_to_id,
                relation_to_id=triples_factory.relation_to_id,
            )
            == triples_factory.mapped_triples
        ).all()

    def test_inverse_triples(self):
        """Test that the right number of entities and triples exist after inverting them."""
        triples_factory = TriplesFactory.from_labeled_triples(triples=triples, create_inverse_triples=True)
        self.assertEqual(4, triples_factory.num_relations)
        self.assertEqual(
            set(range(triples_factory.num_entities)),
            set(triples_factory.entity_to_id.values()),
            msg="wrong number entities",
        )
        self.assertEqual(
            set(range(triples_factory.real_num_relations)),
            set(triples_factory.relation_to_id.values()),
            msg="wrong number relations",
        )

        relations = set(triples[:, 1])
        entities = set(triples[:, 0]).union(triples[:, 2])
        self.assertEqual(len(entities), triples_factory.num_entities, msg="wrong number entities")
        self.assertEqual(2, len(relations), msg="Wrong number of relations in set")
        self.assertEqual(
            2 * len(relations),
            triples_factory.num_relations,
            msg="Wrong number of relations in factory",
        )

    def test_metadata(self):
        """Test metadata passing for triples factories."""
        t = Nations().training
        self.assertEqual(t.metadata, dict(path=NATIONS_TRAIN_PATH))

        entities = ["poland", "ussr"]
        x = t.new_with_restriction(entities=entities)
        entities_ids = t.entities_to_ids(entities=entities)
        self.assertEqual(x.metadata, dict(path=NATIONS_TRAIN_PATH, entity_restriction=entities_ids))

        relations = ["negativebehavior"]
        v = t.new_with_restriction(relations=relations)
        relations_ids = t.relations_to_ids(relations=relations)
        self.assertEqual(v.metadata, dict(path=NATIONS_TRAIN_PATH, relation_restriction=relations_ids))

        w = t.clone_and_exchange_triples(t.triples[0:5], keep_metadata=False)
        self.assertIsInstance(w, TriplesFactory)
        self.assertEqual(w.metadata, dict())

        y, z = t.split()
        self.assertEqual(y.metadata, dict(path=NATIONS_TRAIN_PATH))
        self.assertEqual(z.metadata, dict(path=NATIONS_TRAIN_PATH))

    def test_triples_numeric_literals_factory_split(self):
        """Test splitting a TriplesNumericLiteralsFactory object."""
        # Slightly larger number of triples to guarantee split can find coverage of all entities and relations.
        triples_larger = np.array(
            [
                ["peter", "likes", "chocolate_cake"],
                ["chocolate_cake", "isA", "dish"],
                ["susan", "likes", "chocolate_cake"],
                ["susan", "likes", "pizza"],
                ["peter", "likes", "susan"],
                ["peter", "isA", "person"],
                ["susan", "isA", "person"],
            ],
            dtype=str,
        )

        triples_numeric_literal_factory = TriplesNumericLiteralsFactory.from_labeled_triples(
            triples=triples_larger,
            numeric_triples=numeric_triples,
        )

        left, right = triples_numeric_literal_factory.split()

        self.assertIsInstance(left, TriplesNumericLiteralsFactory)
        self.assertIsInstance(right, TriplesNumericLiteralsFactory)

        assert (left.numeric_literals == triples_numeric_literal_factory.numeric_literals).all()
        assert (right.numeric_literals == triples_numeric_literal_factory.numeric_literals).all()


class TestUtils(unittest.TestCase):
    """Test triples utilities."""

    def test_load_triples_remapped(self):
        """Test loading a triples file where the columns must be remapped."""
        path = os.path.join(RESOURCES, "test_remap.tsv")

        with self.assertRaises(ValueError):
            load_triples(path, column_remapping=[1, 2])

        _triples = load_triples(path, column_remapping=[0, 2, 1])
        self.assertEqual(
            [
                ["a", "r1", "b"],
                ["b", "r2", "c"],
            ],
            _triples.tolist(),
        )

    def test_load_triples_with_nans(self):
        """Test loading triples that have a ``nan`` string.

        .. seealso:: https://github.com/pykeen/pykeen/pull/883
        """
        path = RESOURCES.joinpath("test_nans.tsv")
        expected_triples = [
            ["netherlands", "militaryalliance", "uk"],
            ["egypt", "intergovorgs3", "usa"],
            ["jordan", "relbooktranslations", "nan"],
        ]
        _triples = load_triples(path).tolist()
        self.assertEqual(expected_triples, _triples)

    def test_labeled_binary(self):
        """Test binary i/o on labeled triples factory."""
        tf1 = Nations().training
        self.assert_binary_io(tf1, TriplesFactory)

    def test_core_binary(self):
        """Test binary i/o on core triples factory."""
        tf1 = Nations().training.to_core_triples_factory()
        self.assert_binary_io(tf1, CoreTriplesFactory)

    def assert_binary_io(self, tf, tf_cls):
        """Check the triples factory can be written and reloaded properly."""
        self.assertIsInstance(tf, tf_cls)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            # serialize
            tf.to_path_binary(path)
            # de-serialize
            tf2 = tf_cls.from_path_binary(path)
            # check for equality
            self.assert_tf_equal(tf, tf2)

    def assert_tf_equal(self, tf1, tf2) -> None:
        """Check two triples factories have all of the same stuff."""
        # TODO: this could be (Core)TriplesFactory.__equal__
        self.assertEqual(type(tf1), type(tf2))
        if isinstance(tf1, TriplesFactory):
            self.assertEqual(tf1.entity_labeling, tf2.entity_labeling)
            self.assertEqual(tf1.relation_labeling, tf2.relation_labeling)
        self.assertEqual(tf1.metadata, tf2.metadata)
        self.assertEqual(tf1.create_inverse_triples, tf2.create_inverse_triples)
        self.assertEqual(
            tf1.mapped_triples.detach().cpu().numpy().tolist(),
            tf2.mapped_triples.detach().cpu().numpy().tolist(),
        )
