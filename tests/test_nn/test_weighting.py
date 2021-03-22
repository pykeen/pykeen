# -*- coding: utf-8 -*-

"""Tests for edge weightings."""

import pykeen.nn.weighting
from tests import cases


class InverseInDegreeEdgeWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for inverse in-degree weighting."""

    cls = pykeen.nn.weighting.InverseInDegreeEdgeWeighting


class InverseOutDegreeEdgeWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for inverse out-degree weighting."""

    cls = pykeen.nn.weighting.InverseOutDegreeEdgeWeighting


class SymmetricEdgeWeightingTests(cases.EdgeWeightingTestCase):
    """Tests for symmetric weighting."""

    cls = pykeen.nn.weighting.SymmetricEdgeWeighting
