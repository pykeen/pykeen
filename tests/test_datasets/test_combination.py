"""Tests for graph combination methods."""
import torch
import unittest_templates

import pykeen.datasets.ea.combination
import pykeen.triples.generation
from tests import cases


class DisjointGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for disjoint graph combination."""

    cls = pykeen.datasets.ea.combination.DisjointGraphPairCombinator


class ExtraRelationGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for extra relation graph combination."""

    cls = pykeen.datasets.ea.combination.ExtraRelationGraphPairCombinator


class CollapseGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for collapse graph combination."""

    cls = pykeen.datasets.ea.combination.CollapseGraphPairCombinator


class SwapGraphPairCombinatorTestCase(cases.GraphPairCombinatorTestCase):
    """Tests for swap graph combination."""

    cls = pykeen.datasets.ea.combination.SwapGraphPairCombinator


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
