"""Tests for combination modules."""
import unittest_templates

import pykeen.nn.combinations
from tests import cases


class ConcatCombinationTest(cases.CombinationTestCase):
    """Test for concat combination."""

    cls = pykeen.nn.combinations.ConcatCombination


class CombinationMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.combinations.Combination]):
    """Test for tests for combinations."""

    base_cls = pykeen.nn.combinations.Combination
    base_test = cases.CombinationTestCase
