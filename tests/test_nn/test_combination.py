"""Tests for combination modules."""
import unittest_templates

import pykeen.nn.combination
from tests import cases


class ConcatCombinationTest(cases.CombinationTestCase):
    """Test for concat combination."""

    cls = pykeen.nn.combination.ConcatCombination


class ConcatAggregationCombinationTest(cases.CombinationTestCase):
    """Test for concat + aggregation combination."""

    cls = pykeen.nn.combination.ConcatAggregationCombination


class ConcatProjectionCombinationTest(cases.CombinationTestCase):
    """Test for concat + projection combination."""

    cls = pykeen.nn.combination.ConcatProjectionCombination


class ComplexSeparatedCombinationTest(cases.CombinationTestCase):
    """Test for complex literal combination."""

    cls = pykeen.nn.combination.ComplexSeparatedCombination


class GatedCombinationTest(cases.CombinationTestCase):
    """Test for gated combination."""

    cls = pykeen.nn.combination.GatedCombination


class CombinationMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.combination.Combination]):
    """Test for tests for combinations."""

    base_cls = pykeen.nn.combination.Combination
    base_test = cases.CombinationTestCase
