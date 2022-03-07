"""Tests for training instances."""
from typing import Any, MutableMapping

import torch.utils.data
import unittest_templates

from pykeen.datasets import Nations
from pykeen.triples import LCWAInstances, SLCWAInstances
from pykeen.triples.instances import BatchedSLCWAInstances
from tests import cases


class LCWAInstancesTestCase(cases.TrainingInstancesTestCase):
    """Tests for LCWA training instances."""

    cls = LCWAInstances

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        other_instance = self.cls.from_triples(
            mapped_triples=self.factory.mapped_triples,
            num_entities=self.factory.num_entities,
            num_relations=self.factory.num_relations,
        )
        kwargs["pairs"] = other_instance.pairs
        kwargs["compressed"] = other_instance.compressed
        return kwargs


class SLCWAInstancesTestCase(cases.TrainingInstancesTestCase):
    """Tests for sLCWA training instances."""

    cls = SLCWAInstances

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["mapped_triples"] = self.factory.mapped_triples
        return kwargs


class BatchedSLCWAInstancesTestCase(unittest_templates.GenericTestCase[BatchedSLCWAInstances]):
    """Tests for batched sLCWA training instances."""

    cls = BatchedSLCWAInstances
    batch_size = 2
    kwargs = dict(
        batch_size=batch_size,
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        self.factory = Nations().training
        kwargs["mapped_triples"] = self.factory.mapped_triples
        return kwargs

    def test_data_loader(self):
        """Test data loader."""
        for batch in torch.utils.data.DataLoader(dataset=self.instance, batch_size=None):
            assert batch is not None
