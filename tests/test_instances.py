"""Tests for training instances."""

from collections.abc import MutableMapping
from typing import Any

import numpy

from pykeen.triples import LCWAInstances, SLCWAInstances
from pykeen.triples.instances import BatchedSLCWAInstances, LCWABatch, SLCWABatch, SubGraphSLCWAInstances
from tests import cases


class LCWAInstancesTestCase(cases.TrainingInstancesTestCase):
    """Tests for LCWA training instances."""

    cls = LCWAInstances

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        other_instance = LCWAInstances.from_triples_factory(tf=self.factory)
        kwargs["pairs"] = other_instance.pairs
        kwargs["compressed"] = other_instance.compressed
        return kwargs

    def test_getitem(self) -> None:
        """Test item access."""
        self.instance: LCWAInstances
        item = self.instance[0]
        assert isinstance(item, dict)
        assert {"pairs", "target"}.issubset(item.keys())
        assert item["pairs"].shape == (2,)
        assert item["target"].shape == (self.factory.num_entities,)

    def test_construction(self) -> None:
        """Test proper construction."""
        self.instance: LCWAInstances
        # unique pairs
        assert len(self.instance.pairs) == len(numpy.unique(self.instance.pairs, axis=0))
        # TODO: we could check whether can recover all mapped_triples from pairs + targets


class SLCWAInstancesTestCase(cases.TrainingInstancesTestCase):
    """Tests for sLCWA training instances."""

    cls = SLCWAInstances

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["mapped_triples"] = self.factory.mapped_triples
        return kwargs

    def test_getitem(self) -> None:
        """Test item access."""
        self.instance: SLCWAInstances
        item = self.instance[0]
        assert {"positives", "negatives"}.issubset(item.keys())
        assert item["positives"].shape == (3,)
        assert item["negatives"].shape == (self.instance.sampler.num_negs_per_pos, 3)
        if "pos_weights" in item:
            assert item["pos_weights"].shape == item["positives"].shape
        if "neg_weights" in item:
            assert item["neg_weights"].shape == item["negatives"].shape


class BatchedSLCWAInstancesTestCase(cases.BatchSLCWATrainingInstancesTestCase):
    """Tests for batched sLCWA training instances."""

    cls = BatchedSLCWAInstances


class SubGraphSLCWAInstancesTestCase(cases.BatchSLCWATrainingInstancesTestCase):
    """Tests for subgraph sLCWA training instances."""

    cls = SubGraphSLCWAInstances
