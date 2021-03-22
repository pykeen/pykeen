"""Tests for edge weightings."""

import pykeen.nn.weighting
from tests.cases import EdgeWeightingTestCase


class InverseInDegreeEdgeWeightingTests(EdgeWeightingTestCase):
    """Tests for inverse in-degree weighting."""

    cls = pykeen.nn.weighting.InverseInDegreeEdgeWeighting


class InverseOutDegreeEdgeWeightingTests(EdgeWeightingTestCase):
    """Tests for inverse out-degree weighting."""

    cls = pykeen.nn.weighting.InverseOutDegreeEdgeWeighting


class SymmetricEdgeWeightingTests(EdgeWeightingTestCase):
    """Tests for symmetric weighting."""

    cls = pykeen.nn.weighting.SymmetricEdgeWeighting
