"""Tests for graph combination methods."""
import unittest_templates

import pykeen.datasets.ea.combination
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
