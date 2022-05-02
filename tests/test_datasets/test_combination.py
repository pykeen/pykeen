"""Tests for graph combination methods."""

import torch
import unittest_templates

import pykeen.datasets.ea.combination
import pykeen.triples.generation
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import triple_tensor_to_set
from tests import cases


class DisjointGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for disjoint graph combination."""

    cls = pykeen.datasets.ea.combination.DisjointGraphPairCombinator

    # docstr-coverage: inherited
    def _verify_manual(self, combined_tf: CoreTriplesFactory):  # noqa: D102
        # assumes deterministic entity to id mapping
        expected_triples = {
            # from left_tf
            (0, 0, 1),
            (0, 1, 3),
            (1, 0, 2),
            (4, 1, 5),
            (6, 1, 5),
            # from right_tf with offset
            (7, 2, 8),
            (7, 2, 9),
            (9, 3, 10),
            (11, 3, 12),
            (13, 3, 12),
        }
        assert expected_triples == triple_tensor_to_set(combined_tf.mapped_triples)


class ExtraRelationGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for extra relation graph combination."""

    cls = pykeen.datasets.ea.combination.ExtraRelationGraphPairCombinator
    same_as_rel_name = cls.ALIGNMENT_RELATION_NAME

    # docstr-coverage: inherited
    def _verify_manual(self, combined_tf: CoreTriplesFactory):  # noqa: D102
        same_as_id = combined_tf.relation_to_id[self.__class__.same_as_rel_name]
        assert isinstance(same_as_id, int)
        # assumes deterministic entity to id mapping
        expected_triples = {
            # from left_tf
            (0, 0, 1),
            (0, 1, 3),
            (1, 0, 2),
            (4, 1, 5),
            (6, 1, 5),
            # from right_tf with offset
            (7, 2, 8),
            (7, 2, 9),
            (9, 3, 10),
            (11, 3, 12),
            (13, 3, 12),
            # extra-relation
            (3, same_as_id, 10),
            (4, same_as_id, 11),
            (5, same_as_id, 12),
            (6, same_as_id, 13),
        }
        assert expected_triples == triple_tensor_to_set(combined_tf.mapped_triples)


class CollapseGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for collapse graph combination."""

    cls = pykeen.datasets.ea.combination.CollapseGraphPairCombinator

    # docstr-coverage: inherited
    def _verify_manual(self, combined_tf: CoreTriplesFactory):  # noqa: D102
        # assumes deterministic entity to id mapping
        expected_triples = {
            (0, 0, 1),
            (0, 1, 3),
            (1, 0, 2),
            (4, 1, 5),
            (6, 1, 5),
            (7, 2, 8),
            (7, 2, 9),
            (9, 3, 3),
            (4, 3, 5),
            (6, 3, 5),
        }
        assert expected_triples == triple_tensor_to_set(combined_tf.mapped_triples)


class SwapGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for swap graph combination."""

    cls = pykeen.datasets.ea.combination.SwapGraphPairCombinator

    # docstr-coverage: inherited
    def _verify_manual(self, combined_tf: CoreTriplesFactory):  # noqa: D102
        # assumes deterministic entity to id mapping
        expected_triples = {
            # from left_tf
            (0, 0, 1),
            (0, 1, 3),
            (1, 0, 2),
            (4, 1, 5),
            (6, 1, 5),
            # from right_tf with offset
            (7, 2, 8),
            (7, 2, 9),
            (9, 3, 10),
            (11, 3, 12),
            (13, 3, 12),
            # additional
            (11, 1, 5),
            (13, 1, 5),
            (4, 3, 12),
            (6, 3, 12),
            (0, 1, 10),
            (4, 1, 12),
            (6, 1, 12),
            (9, 3, 3),
            (11, 3, 5),
            (13, 3, 5),
        }
        assert expected_triples == triple_tensor_to_set(combined_tf.mapped_triples)


class GraphPairCombinatorMetaTestCase(
    unittest_templates.MetaTestCase[pykeen.datasets.ea.combination.GraphPairCombinator]
):
    """Test for tests for graph combination methods."""

    base_cls = pykeen.datasets.ea.combination.GraphPairCombinator
    base_test = cases.GraphPairCombinatorTestCase


def test_cat_shift_triples():
    """Test cat_shift_triples."""
    first, second = pykeen.triples.generation.generate_triples(), pykeen.triples.generation.generate_triples()
    combined, offsets = pykeen.datasets.ea.combination.cat_shift_triples(first, second)
    # verify shape
    assert combined.shape == (first.shape[0] + second.shape[0], 3)
    assert offsets.shape == (2, 2)
    # verify dtype
    assert combined.dtype == first.dtype
    assert offsets.dtype == torch.long
    # verify number of entities/relations
    num_entities_first = first[:, 0::2].max().item() + 1
    num_relations_first = first[:, 1].max().item() + 1
    combined_num_entities = num_entities_first + second[:, 0::2].max().item() + 1
    combined_num_relations = num_relations_first + second[:, 1].max().item() + 1
    assert combined[:, 0::2].max().item() == combined_num_entities - 1
    assert combined[:, 1].max().item() == combined_num_relations - 1
    # verify offsets
    exp_offsets = torch.as_tensor([[0, 0], [num_entities_first, num_relations_first]])
    assert (offsets == exp_offsets).all()


def test_iter_entity_mappings():
    """Test iter_entity_mappings."""
    # create old, new pairs
    max_id = 33
    # simulate merging ids
    old = torch.arange(max_id)
    new = torch.randint(max_id, size=(max_id,)).sort().values
    pairs = [(old, new)]
    # only a single pair
    offsets = torch.as_tensor([0])
    # apply
    mappings = list(pykeen.datasets.ea.combination.iter_entity_mappings(*pairs, offsets=offsets))
    assert len(mappings) == len(pairs)
    for (old, new), mapi in zip(pairs, mappings):
        # every key is contained
        assert set(mapi.keys()) == set(old.tolist())
        # value range
        assert set(mapi.values()) == set(new.tolist())
