"""Test the PyKEEN sample weighters."""

import torch
import unittest_templates

import pykeen.triples.weights
from tests import cases


class RelationFrequencyLossWeighterTests(cases.LossWeightTestCase):
    """Unit test for relation sample weighter."""

    cls = pykeen.triples.weights.RelationLossWeighter

    def _pre_instantiation_hook(
        self, kwargs: cases.MutableMapping[str, cases.Any]
    ) -> cases.MutableMapping[str, cases.Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["weights"] = torch.rand(size=(self.num_relations,), generator=self.generator)
        return kwargs


class TestLossWeighters(unittest_templates.MetaTestCase[pykeen.triples.weights.LossWeighter]):
    """Test that the loss weighters all have tests."""

    base_cls = pykeen.triples.weights.LossWeighter
    base_test = cases.LossWeightTestCase
