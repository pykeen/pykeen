"""Tests for training instances."""
from typing import Any, MutableMapping

from pykeen.triples import LCWAInstances, SLCWAInstances
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
