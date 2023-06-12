# -*- coding: utf-8 -*-

"""Unit tests for quadruples factories."""

import itertools as itt
import unittest
from typing import Collection, Optional
from unittest.mock import patch

import numpy as np
import torch

from pykeen.datasets.temporal import SmallSample
from pykeen.triples import LCWAQuadrupleInstances, QuadruplesFactory
from pykeen.triples.triples_factory import INVERSE_SUFFIX, QUADRUPLES_DF_COLUMNS
from utils import needs_packages

quadruples = np.array(
    [
        ["peter", "likes", "chocolate_cake", "2019-01-01"],
        ["chocolate_cake", "isA", "dish", "2021-03-01"],
        ["susan", "likes", "pizza", "2023-09-26"],
        ["peter", "likes", "susan", "1999-02-14"],
    ],
    dtype=str,
)

quadruples_with_nans = [
    ["nan", "hates", "banana", "2001-08-29"],
    ["susan", "nan", "dancing", "2010-09-10"],
    ["apple", "isA", "nan", "2000-01-01"],
    ["peter", "likes", "basketball", "nan"],
]


class TestQuadruplesFactory(unittest.TestCase):
    """Class for testing quadruples factories."""

    def setUp(self) -> None:
        """Instantiate test instance."""
        self.factory = SmallSample().training

    def test_correct_inverse_creation(self) -> None:
        """Test if the quadruples and the corresponding inverses are created."""
        q = [
            ["e1", "a.", "e5", "2019-01-01"],
            ["e1", "a", "e2", "2021-03-01"],
        ]
        q = np.array(q, dtype=str)
        factory = QuadruplesFactory.from_labeled_quadruples(
            quadruples=q, create_inverse_quadruples=True
        )
        
        instances = factory.create_slcwa_instances()
        assert len(instances) == 4

    def test_automatic_incomplete_inverse_detection(self):
        """Test detecting that the quadruples contain inverses, warns about them, and filters them out."""
        # comment(mberr): from my pov this behaviour is faulty: the quadruples factory is expected to say it contains
        # inverse relations, although the quadruples contained in it are not the same we would have when removing the
        # first quadruple, and passing create_inverse_quadruples=True.
        q = [
            ["e3", f"a.{INVERSE_SUFFIX}", "e10", "2023-01-11"],
            ["e1", "a", "e2", "1978-03-27"],
            ["e1", "a.", "e5", "2019-01-01"],
        ]
        q = np.array(q, dtype=str)
        for create_inverse_quadruples in (False, True):
            with patch("pykeen.triples.triples_factory.logger.warning") as warning:
                factory = QuadruplesFactory.from_labeled_quadruples(
                    quadruples=q, create_inverse_quadruples=create_inverse_quadruples
                )
                # check for warning
                warning.assert_called()
                # check for filtered quadruples
                assert factory.num_triples == 2
                # check for correct inverse quadruples flag
                assert factory.create_inverse_triples == create_inverse_quadruples

    def test_id_to_label(self):
        """Test ID-to-label conversion."""
        for label_to_id, id_to_label in [
            (self.factory.entity_to_id, self.factory.entity_id_to_label),
            (self.factory.relation_to_id, self.factory.relation_id_to_label),
            (self.factory.timestamp_to_id, self.factory.timestamp_id_to_label),
        ]:
            for k in label_to_id.keys():
                assert id_to_label[label_to_id[k]] == k
            for k in id_to_label.keys():
                assert label_to_id[id_to_label[k]] == k

    def test_tensor_to_df(self):
        """Test tensor_to_df()."""
        # check correct translation
        labeled_quadruples = set(tuple(row) for row in self.factory.quadruples.tolist())
        tensor = self.factory.mapped_quadruples
        scores = torch.rand(tensor.shape[0])
        df = self.factory.tensor_to_df(tensor=tensor, scores=scores)
        re_labeled_quadruples = set(
            tuple(row)
            for row in df[
                ["head_label", "relation_label", "tail_label", "timestamp_label"]
            ].values.tolist()
        )
        assert labeled_quadruples == re_labeled_quadruples

        # check column order
        assert tuple(df.columns) == QUADRUPLES_DF_COLUMNS + ("scores",)

    def _test_restriction(
        self,
        original_quadruples_factory: QuadruplesFactory,
        entity_restriction: Optional[Collection[str]],
        invert_entity_selection: bool,
        relation_restriction: Optional[Collection[str]],
        invert_relation_selection: bool,
        timestamp_restriction: Optional[Collection[str]],
        invert_timestamp_selection: bool,
    ):
        """Run the actual test for new_with_restriction."""
        # apply restriction

        restricted_quadruples_factory = original_quadruples_factory.new_with_restriction(
            entities=entity_restriction,
            relations=relation_restriction,
            timestamps=timestamp_restriction,
            invert_entity_selection=invert_entity_selection,
            invert_relation_selection=invert_relation_selection,
            invert_timestamp_selection=invert_timestamp_selection,
        )

        # check that the quadruples factory is returned as is, if and only if no restriction is to apply
        no_restriction_to_apply = (
            entity_restriction is None
            and relation_restriction is None
            and timestamp_restriction is None
        )
        equal_factory_object = id(restricted_quadruples_factory) == id(original_quadruples_factory)
        assert no_restriction_to_apply == equal_factory_object

        # check that inverse_quadruples is correctly carried over
        assert (
            original_quadruples_factory.create_inverse_triples
            == restricted_quadruples_factory.create_inverse_triples
        )

        # verify that the label-to-ID mapping has not been changed
        assert (
            original_quadruples_factory.entity_to_id 
            == restricted_quadruples_factory.entity_to_id
        )
        assert (
            original_quadruples_factory.relation_to_id
            == restricted_quadruples_factory.relation_to_id
        )
        assert (
            original_quadruples_factory.timestamp_to_id
            == restricted_quadruples_factory.timestamp_to_id
        )

        # verify that quadruples have been filtered
        if entity_restriction is not None:
            present_entities = set(restricted_quadruples_factory.quadruples[:, 0]).union(
                restricted_quadruples_factory.quadruples[:, 2]
            )
            expected_entities = (
                set(original_quadruples_factory.entity_id_to_label.values()).difference(
                    entity_restriction
                )
                if invert_entity_selection
                else entity_restriction
            )
            assert expected_entities.issuperset(present_entities)

        if relation_restriction is not None:
            present_relations = set(restricted_quadruples_factory.quadruples[:, 1])
            expected_relations = (
                set(original_quadruples_factory.relation_id_to_label.values())
                if invert_relation_selection
                else set(relation_restriction)
            )
            assert expected_relations.issuperset(present_relations)

        if timestamp_restriction is not None:
            present_timestamps = set(restricted_quadruples_factory.quadruples[:, 3])
            """
            expected_timestamps = (
                set(original_quadruples_factory.timestamp_id_to_label.values())
                if invert_timestamp_selection
                else set(timestamp_restriction)
            )
            """
            if invert_timestamp_selection:
                expected_timestamps = set(original_quadruples_factory.timestamp_id_to_label.values())
            else:
                expected_timestamps = set(timestamp_restriction)

            assert expected_timestamps.issuperset(present_timestamps)

    def test_new_with_restriction(self):
        """Test new_with_restriction()."""
        relation_restrictions = {
            "1",
            "2",
        }
        entity_restrictions = {
            "0",
            "1",
        }
        timestamp_restrictions = {
            "2000-01-01", 
            "2000-01-02"
        }
        for inverse_quadruples in (True, False):
            original_quadruples_factory = SmallSample(
                create_inverse_quadruples=inverse_quadruples,
            ).training

            # Test different combinations of restrictions
            for (
                (entity_restriction, invert_entity_selection),
                (relation_restriction, invert_relation_selection),
                (timestamp_restriction, invert_timestamp_selection),
            ) in itt.product(
                ((None, None), (entity_restrictions, False), (entity_restrictions, True)),
                ((None, None), (relation_restrictions, False), (relation_restrictions, True)),
                ((None, None), (timestamp_restrictions, False), (timestamp_restrictions, True)),
            ):
                with self.subTest(
                    entity_restriction=entity_restriction,
                    invert_entity_selection=invert_entity_selection,
                    relation_restriction=relation_restriction,
                    invert_relation_selection=invert_relation_selection,
                    timestamp_restriction=timestamp_restriction,
                    invert_timestamp_selection=invert_timestamp_selection,
                ):
                    self._test_restriction(
                        original_quadruples_factory=original_quadruples_factory,
                        entity_restriction=entity_restriction,
                        invert_entity_selection=invert_entity_selection,
                        relation_restriction=relation_restriction,
                        invert_relation_selection=invert_relation_selection,
                        timestamp_restriction=timestamp_restriction,
                        invert_timestamp_selection=invert_timestamp_selection,
                    )

    def test_create_lcwa_instances(self):
        """Test create_lcwa_instances."""
        factory = SmallSample().training
        instances = factory.create_lcwa_instances()
        assert isinstance(instances, LCWAQuadrupleInstances)

        # check compressed quadruples
        # reconstruct quadruples from compressed form
        reconstructed_quadruples = set()
        for hrtime, row_id in zip(instances.pairs, range(instances.compressed.shape[0])):
            h, r, time = hrtime.tolist()
            _, tails = instances.compressed[row_id].nonzero()
            reconstructed_quadruples.update((h, r, t, time) for t in tails.tolist())
        original_quadruples = {tuple(hrttime) for hrttime in factory.mapped_quadruples.tolist()}
        assert original_quadruples == reconstructed_quadruples

        # check data loader
        for batch in torch.utils.data.DataLoader(instances, batch_size=2):
            assert len(batch) == 2
            assert all(torch.is_tensor(x) for x in batch)
            x, y = batch
            batch_size = x.shape[0]
            assert x.shape == (batch_size, 3)
            assert x.dtype == torch.long
            assert y.shape == (batch_size, factory.num_entities)
            assert y.dtype == torch.get_default_dtype()

    def test_split_inverse_quadruples(self):
        """Test whether inverse quadruples are only created in the training factory."""
        # set create inverse quadruple to true
        self.factory.create_inverse_triples = True
        # split factory
        train, *others = self.factory.split()
        # check that in *training* inverse quadruples are to be created
        assert train.create_inverse_triples
        # check that in all other splits no inverse quadruples are to be created
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
    
    @needs_packages("wordcloud", "IPython")
    def test_timestamp_word_cloud(self):
        """Test word cloud generation."""
        wc = self.factory.timestamp_word_cloud(top=3)
        self.assertIsNotNone(wc)

if __name__ == "__main__":
    test = TestQuadruplesFactory()
    test.setUp()
    test.test_correct_inverse_creation()
    test.test_automatic_incomplete_inverse_detection()
    test.test_id_to_label()
    test.test_new_with_restriction()
    test.test_create_lcwa_instances()
    test.test_split_inverse_quadruples()
    test.test_entity_word_cloud()
    test.test_relation_word_cloud()
    test.test_timestamp_word_cloud()
