"""Tests for combination modules."""
from typing import Sequence, Tuple

import torch
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
    kwargs = dict(input_dims=cases.CombinationTestCase.input_dims[0])
    input_dims = cases.CombinationTestCase.input_dims[0:1]


class ComplexSeparatedCombinationTest(cases.CombinationTestCase):
    """Test for complex literal combination."""

    cls = pykeen.nn.combination.ComplexSeparatedCombination

    def _create_input(self, input_shapes: Sequence[Tuple[int, ...]]) -> Sequence[torch.FloatTensor]:
        # requires at least one complex tensor as input
        first_shape, *input_shapes = input_shapes
        return torch.empty(size=first_shape, dtype=torch.cfloat), *super()._create_input(input_shapes=input_shapes)


class GatedCombinationTest(cases.CombinationTestCase):
    """Test for gated combination."""

    cls = pykeen.nn.combination.GatedCombination
    kwargs = dict(
        entity_dim=cases.CombinationTestCase.input_dims[0][0],
        literal_dim=cases.CombinationTestCase.input_dims[0][1],
    )
    input_dims = cases.CombinationTestCase.input_dims[0:1]


class CombinationMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.combination.Combination]):
    """Test for tests for combinations."""

    base_cls = pykeen.nn.combination.Combination
    base_test = cases.CombinationTestCase
