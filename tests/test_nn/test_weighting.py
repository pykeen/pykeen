"""Tests for edge weightings."""

import unittest_templates

import pykeen.nn.weighting
from tests import cases

from ..utils import needs_packages


class InverseInDegreeEdgeWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for inverse in-degree weighting."""

    cls = pykeen.nn.weighting.InverseInDegreeEdgeWeighting


class InverseOutDegreeEdgeWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for inverse out-degree weighting."""

    cls = pykeen.nn.weighting.InverseOutDegreeEdgeWeighting


class SymmetricEdgeWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for symmetric weighting."""

    cls = pykeen.nn.weighting.SymmetricEdgeWeighting


@needs_packages("torch_scatter")
class AttentionWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for attention weighting."""

    cls = pykeen.nn.weighting.AttentionEdgeWeighting
    # message_dim must be divisible by num_heads
    message_dim = 4
    kwargs = dict(
        message_dim=4,
        num_heads=2,
    )


class WeightingMetaTestCase(unittest_templates.MetaTestCase[pykeen.nn.weighting.EdgeWeighting]):
    """Tests for weighting test coverage."""

    base_cls = pykeen.nn.weighting.EdgeWeighting
    base_test = cases.EdgeWeightingTestCase
