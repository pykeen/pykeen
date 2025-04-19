"""Test the PyKEEN sample weighters."""

import unittest_templates

import pykeen.triples.weights
from tests import cases


class RelationFrequencyLossWeighterTests(cases.LossWeightTestCase):
    """Unit test for relation sample weighter."""

    cls = pykeen.triples.weights.RelationSampleWeighter


class TestLossWeighters(unittest_templates.MetaTestCase[pykeen.triples.weights.SampleWeighter]):
    """Test that the loss weighters all have tests."""

    base_cls = pykeen.triples.weights.SampleWeighter
    base_test = cases.LossWeightTestCase
