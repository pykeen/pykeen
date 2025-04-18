"""Unit tests for triples factories."""

import itertools as itt
import os
import tempfile
import unittest
from collections.abc import Collection, Iterable, Mapping
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pykeen.datasets import Hetionet, Nations, SingleTabbedDataset
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from pykeen.triples import CoreTriplesFactory, TriplesFactory, TriplesNumericLiteralsFactory, generation
from pykeen.triples.splitting import splitter_resolver
from pykeen.triples.triples_factory import (
    INVERSE_SUFFIX,
    _map_triples_elements_to_ids,
    get_mapped_triples,
    valid_triple_id_range,
)
from pykeen.triples.utils import TRIPLES_DF_COLUMNS, load_triples
from tests.constants import RESOURCES
from tests.utils import needs_packages

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
        entity_restriction: Collection[str] | None,
        invert_entity_selection: bool,
        relation_restriction: Collection[str] | None,
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

    @needs_packages("wordcloud", "IPython")
    def test_entity_word_cloud(self):
        """Test word cloud generation."""
        wc = self.factory.entity_word_cloud(top=3)
        self.assertIsNotNone(wc)

    @needs_packages("wordcloud", "IPython")
    def test_relation_word_cloud(self):
        """Test word cloud generation."""
        wc = self.factory.relation_word_cloud(top=3)
        self.assertIsNotNone(wc)


class TestSplit(unittest.TestCase):
    """Test splitting."""

    triples_factory: TriplesFactory

    def setUp(self) -> None:
        """Set up the tests."""
        self.dataset = Nations()
        self.triples_factory = self.dataset.training
        self.assertEqual(1592, self.triples_factory.num_triples)

    def _test_invariants_shared(self, *factories: TriplesFactory, lossy: bool = False) -> None:
        # verify that the type got correctly promoted
        for factory in factories:
            self.assertEqual(type(factory), type(self.triples_factory))
            # we only support inductive *entity* splits for now
            self.assertEqual(factory.num_relations, self.triples_factory.num_relations)
            # verify that triple have been compacted
            self.assertTrue(
                valid_triple_id_range(
                    factory.mapped_triples, num_entities=factory.num_entities, num_relations=factory.num_relations
                )
            )
        # verify that no triple got lost
        total_num_triples = sum(t.num_triples for t in factories)
        if lossy:
            self.assertLessEqual(total_num_triples, self.triples_factory.num_triples)
        else:
            self.assertEqual(total_num_triples, self.triples_factory.num_triples)

    def _test_invariants_transductive(
        self, training_triples_factory: TriplesFactory, *other_factories: TriplesFactory, lossy: bool = False
    ) -> None:
        """Test invariants for result of triples factory splitting."""
        # verify that all entities and relations are present in the training factory
        self.assertEqual(training_triples_factory.num_entities, self.triples_factory.num_entities)

        all_factories = (training_triples_factory, *other_factories)
        self._test_invariants_shared(*all_factories, lossy=lossy)

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
                self._test_invariants_transductive(*factories_1)

                factories_2 = self.triples_factory.split(ratios, method=method, random_state=0)
                self.assertEqual(n, len(factories_2))
                self._test_invariants_transductive(*factories_2)

                self._compare_factories(factories_1, factories_2)

    def test_semi_inductive_split(self) -> None:
        """Test semi-inductive splitting."""
        cases = [
            (2, 0.8),
            (2, [0.8]),
            (3, [0.80, 0.10]),
            (3, [0.80, 0.10, 0.10]),
        ]
        for n, ratios in cases:
            with self.subTest(ratios=ratios):
                factories_1 = self.triples_factory.split_semi_inductive(ratios, random_state=0)
                self.assertEqual(n, len(factories_1))
                self._test_invariants_transductive(*factories_1, lossy=True)
                # TODO: there are other invariants to check than for transductive splits

                # check for reproducibility, by splitting a second time with the same seed
                factories_2 = self.triples_factory.split_semi_inductive(ratios, random_state=0)
                self._compare_factories(factories_1, factories_2)

    def test_fully_inductive_split(self) -> None:
        """Test semi-inductive splitting."""
        cases = [
            (3, 0.5, 0.8),
            (3, 0.5, [0.8]),
            (4, 0.6, [0.80, 0.10]),
            (4, 0.4, [0.80, 0.10, 0.10]),
        ]
        for n, entity_split, triple_ratios in cases:
            with self.subTest(entity_split=entity_split, triple_ratios=triple_ratios):
                factories_1 = self.triples_factory.split_fully_inductive(
                    entity_split_train_ratio=entity_split, evaluation_triples_ratios=triple_ratios, random_state=0
                )
                self.assertEqual(n, len(factories_1))
                # in the fully inductive setting, we have two separate graphs, with all but the training factory
                # in the inference graph.
                self._test_invariants_shared(*factories_1[1:], lossy=True)
                # TODO: there are other invariants to check than for transductive splits

                # check for reproducibility, by splitting a second time with the same seed
                factories_2 = self.triples_factory.split_fully_inductive(
                    entity_split_train_ratio=entity_split, evaluation_triples_ratios=triple_ratios, random_state=0
                )
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
        for factory_1, factory_2 in zip(factories_1, factories_2, strict=False):
            triples_1 = factory_1.mapped_triples
            triples_2 = factory_2.mapped_triples
            self.assertTrue(torch.equal(triples_1, triples_2), msg=msg)


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

        w = t.clone_and_exchange_triples(t.mapped_triples[0:5], keep_metadata=False)
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

        .. seealso::

            https://github.com/pykeen/pykeen/pull/883
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

    def test_core_binary_inverse_relations(self):
        """Test binary i/o on core triples factory with inverse relations."""
        tf1 = Nations(create_inverse_triples=True).training.to_core_triples_factory()
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
        self.assertEqual(tf1.num_entities, tf2.num_entities)
        self.assertEqual(tf1.num_relations, tf2.num_relations)
        self.assertEqual(tf1.create_inverse_triples, tf2.create_inverse_triples)
        self.assertEqual(
            tf1.mapped_triples.detach().cpu().numpy().tolist(),
            tf2.mapped_triples.detach().cpu().numpy().tolist(),
        )


# cf. https://docs.pytest.org/en/7.1.x/example/parametrize.html#parametrizing-conditional-raising
@pytest.mark.parametrize(
    ["dtype", "size", "expectation"],
    [
        # wrong ndim
        (torch.long, (3,), pytest.raises(ValueError)),
        # wrong last dim
        (torch.long, (3, 11), pytest.raises(ValueError)),
        # wrong dtype: float
        (torch.float, (11, 3), pytest.raises(TypeError)),
        # wrong dtype: complex
        (torch.cfloat, (11, 3), pytest.raises(TypeError)),
        # correct
        (torch.long, (11, 3), does_not_raise()),
        (torch.long, (0, 3), does_not_raise()),
        (torch.uint8, (11, 3), does_not_raise()),
        (torch.bool, (11, 3), does_not_raise()),
    ],
)
def test_core_triples_factory_error_handling(dtype: torch.dtype, size: tuple[int, ...], expectation):
    """Test error handling in init method of CoreTriplesFactory."""
    max_id_upper_bound = 33
    with expectation:
        CoreTriplesFactory(
            mapped_triples=torch.randint(max_id_upper_bound, size=size).to(dtype=dtype),
            num_entities=max_id_upper_bound,
            num_relations=max_id_upper_bound,
        )


def _iter_get_mapped_triples_inputs() -> Iterable[tuple[Any, Mapping[str, Any]]]:
    """Iterate valid test inputs for get_mapped_triples."""
    factory = Nations().training
    # >>> positional argument
    # mapped_triples
    yield factory.mapped_triples, {}
    # triples factory
    yield factory, {}
    # labeled triples + factory
    labeled = [("brazil", "accusation", "burma"), ("brazil", "accusation", "uk")]
    # single labeled triple
    yield labeled[0], dict(factory=factory)
    # multiple labeled triples as list
    yield labeled, dict(factory=factory)
    # multiple labeled triples as array
    yield np.asarray(labeled), dict(factory=factory)
    # >>> keyword only
    yield None, dict(mapped_triples=factory.mapped_triples)
    yield None, dict(factory=factory)
    yield None, dict(triples=labeled, factory=factory)
    yield None, dict(triples=np.asarray(labeled), factory=factory)


@pytest.mark.parametrize(["x", "inputs"], _iter_get_mapped_triples_inputs())
def test_get_mapped_triples(x, inputs: Mapping[str, Any]):
    """Test get_mapped_triples."""
    mapped_triples = get_mapped_triples(x, **inputs)
    assert torch.is_tensor(mapped_triples)
    assert mapped_triples.dtype == torch.long
    assert mapped_triples.ndim == 2
    assert mapped_triples.shape[-1] == 3


@pytest.fixture()
def tf_one_hole() -> CoreTriplesFactory:
    """Create a condensable triples factory."""
    # create (already condensed) triples factory
    tf = generation.generate_triples_factory(random_state=42)
    # remove all triples with entity or relation ID 0
    keep_mask = (tf.mapped_triples != 0).all(dim=-1)
    tf.mapped_triples = tf.mapped_triples[keep_mask]
    return tf


@pytest.mark.parametrize(("entities", "relations"), itt.product((False, True), repeat=2))
def test_condense(tf_one_hole: CoreTriplesFactory, entities: bool, relations: bool) -> None:
    """Test condensation."""
    tf_new = tf_one_hole.condense(entities=entities, relations=relations)
    expected_num_entities = tf_one_hole.num_entities - 1 if entities else tf_one_hole.num_entities
    assert tf_new.num_entities == expected_num_entities
    expected_num_relations = tf_one_hole.num_relations - 1 if relations else tf_one_hole.num_relations
    assert tf_new.num_relations == expected_num_relations
