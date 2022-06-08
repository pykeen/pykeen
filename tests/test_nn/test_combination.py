"""Tests for combination modules."""
import unittest_templates

import pykeen.nn.combination
from tests import cases


class ConcatCombinationTest(cases.CombinationTestCase):
    """Test for concat combination."""

    cls = pykeen.nn.combination.ConcatCombination


class CombinationMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.combination.Combination]):
    """Test for tests for combinations."""

    base_cls = pykeen.nn.combination.Combination
    base_test = cases.CombinationTestCase
