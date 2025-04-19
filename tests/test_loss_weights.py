"""Test the PyKEEN sample weighters."""

import unittest_templates

import pykeen.triples.weights
from tests import cases


class RelationFrequencyLossWeighterTests(cases.LossWeightTestCase):
    """Unit test for relation sample weighter."""

    cls = pykeen.triples.weights.RelationLossWeighter


class TestLossWeighters(unittest_templates.MetaTestCase[pykeen.triples.weights.LossWeighter]):
    """Test that the loss weighters all have tests."""

    base_cls = pykeen.triples.weights.LossWeighter
    base_test = cases.LossWeightTestCase
