"""Tests for training instances."""

from collections.abc import MutableMapping
from typing import Any

import numpy
import pytest
import torch

from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.triples import LCWAInstances
from pykeen.triples.instances import BatchedSLCWAInstances, SubGraphSLCWAInstances
from pykeen.triples.triples_factory import TriplesFactory
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
        factory = TriplesFactory.from_path(NATIONS_TRAIN_PATH)
        instances = LCWAInstances.from_triples_factory(factory)
        assert isinstance(instances, LCWAInstances)

        # check compressed triples
        # reconstruct triples from compressed form
        reconstructed_triples: set[tuple[int, int, int]] = set()
        for hr, row_id in zip(instances.pairs, range(instances.compressed.shape[0]), strict=False):
            h, r = hr.tolist()
            _, tails = instances.compressed[row_id].nonzero()
            reconstructed_triples.update((h, r, t) for t in tails.tolist())
        original_triples = {tuple(hrt) for hrt in factory.mapped_triples.tolist()}
        assert original_triples == reconstructed_triples

        # check data loader
        for batch in torch.utils.data.DataLoader(instances, batch_size=2):
            assert isinstance(batch, dict)  # i.e., a  LCWABatch
            assert {"pairs", "target"} == batch.keys()
            assert torch.is_tensor(batch["pairs"])
            assert torch.is_tensor(batch["target"])

            x, y = batch["pairs"], batch["target"]
            batch_size = x.shape[0]
            assert x.shape == (batch_size, 2)
            assert x.dtype == torch.long
            assert y.shape == (batch_size, factory.num_entities)
            assert y.dtype == torch.get_default_dtype()


class BatchedSLCWAInstancesTestCase(cases.BatchSLCWATrainingInstancesTestCase):
    """Tests for batched sLCWA training instances."""

    cls = BatchedSLCWAInstances

    def test_correct_inverse_creation(self):
        """Test if the triples and the corresponding inverses are created."""
        t = [
            ["e1", "a.", "e5"],
            ["e1", "a", "e2"],
        ]
        t = numpy.array(t, dtype=str)
        factory = TriplesFactory.from_labeled_triples(triples=t, create_inverse_triples=True)
        instances = BatchedSLCWAInstances.from_triples_factory(factory)
        assert len(instances) == 4

    def test_grouped(self):
        """Test that grouped instances emit the expected keys and shapes."""
        instances = BatchedSLCWAInstances.from_triples_factory(
            self.factory,
            batch_size=2,
            negative_sampler_kwargs={"num_negs_per_pos": 3},
            grouped=True,
        )
        for batch in torch.utils.data.DataLoader(dataset=instances, batch_size=None):
            assert isinstance(batch, dict)
            assert {"positives", "corruptions"}.issubset(batch.keys())
            assert batch["positives"].shape == (2, 3)
            assert "negatives" not in batch
            assert sum(replacements.shape[-1] for replacements in batch["corruptions"].values()) == 3
            for replacements in batch["corruptions"].values():
                assert replacements.shape[0] == 2

    def test_grouped_requires_supporting_sampler(self):
        """Test that grouped=True with a non-supporting sampler raises a ValueError."""
        assert not BernoulliNegativeSampler.supports_grouped_corruption
        with pytest.raises(ValueError, match="grouped"):
            BatchedSLCWAInstances.from_triples_factory(
                self.factory,
                negative_sampler=BernoulliNegativeSampler,
                grouped=True,
            )


class SubGraphSLCWAInstancesTestCase(cases.BatchSLCWATrainingInstancesTestCase):
    """Tests for subgraph sLCWA training instances."""

    cls = SubGraphSLCWAInstances
