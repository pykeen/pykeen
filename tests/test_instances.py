"""Tests for training instances."""

import copy
import itertools
from collections.abc import MutableMapping
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.triples import LCWAInstances
from pykeen.triples.instances import BatchedSLCWAInstances, SubGraphSLCWAInstances
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.weights import RelationLossWeighter
from tests import cases


class LCWAInstancesTestCase(cases.TrainingInstancesTestCase):
    """Tests for LCWA training instances."""

    cls = LCWAInstances

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
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
        t = np.array(t, dtype=str)
        factory = TriplesFactory.from_labeled_triples(triples=t, create_inverse_triples=True)
        instances = BatchedSLCWAInstances.from_triples_factory(factory)
        assert len(instances) == 4

    @staticmethod
    def _iter_triple_ids_multiple_workers(
        workers: list[BatchedSLCWAInstances], base_seed: int
    ) -> list[list[list[int]]]:
        """Collect the batches of triple IDs of each simulated data loader worker."""
        result = []
        for worker_id, worker in enumerate(workers):
            worker_info = mock.Mock(id=worker_id, num_workers=len(workers), seed=base_seed + worker_id)
            with mock.patch("torch.utils.data.get_worker_info", return_value=worker_info):
                result.append([[int(i) for i in batch] for batch in worker.iter_triple_ids()])
        return result

    def test_iter_triple_ids_multiple_workers(self):
        """Test that multi-worker iteration yields the same number of batches and triple IDs as single-process."""
        for num_triples, batch_size, num_workers, drop_last in itertools.product(
            (1, 13, 14, 50), (1, 3, 7), (1, 2, 3, 5, 11), (False, True)
        ):
            with self.subTest(
                num_triples=num_triples, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last
            ):
                instance = BatchedSLCWAInstances(
                    mapped_triples=self.factory.mapped_triples[:num_triples],
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
                # each worker process operates on its own copy of the dataset
                workers = [copy.deepcopy(instance) for _ in range(num_workers)]
                # simulate multiple epochs with persistent workers, as well as a new base seed
                for epoch, base_seed in enumerate((42, 42, 43)):
                    batches = [
                        batch
                        for worker_batches in self._iter_triple_ids_multiple_workers(workers, base_seed=base_seed)
                        for batch in worker_batches
                    ]
                    # same number of batches as single-process
                    assert len(batches) == len(instance), epoch
                    # at most one incomplete batch, which is dropped iff drop_last
                    sizes = sorted(len(batch) for batch in batches)
                    assert all(size == batch_size for size in sizes[1:])
                    if drop_last and sizes:
                        assert sizes[0] == batch_size
                    triple_ids = sorted(i for batch in batches for i in batch)
                    # no duplicates
                    assert len(triple_ids) == len(set(triple_ids))
                    assert set(triple_ids).issubset(range(num_triples))
                    if drop_last:
                        assert num_triples - len(triple_ids) < batch_size
                    else:
                        assert triple_ids == list(range(num_triples))

    def test_iter_triple_ids_multiple_workers_randomness(self):
        """Test that the shared permutation is random, and changes between epochs."""
        instance = BatchedSLCWAInstances(mapped_triples=self.factory.mapped_triples, batch_size=7, drop_last=False)
        workers = [copy.deepcopy(instance) for _ in range(3)]
        first, second, third = (
            self._iter_triple_ids_multiple_workers(workers, base_seed=base_seed) for base_seed in (42, 42, 43)
        )
        assert first != second  # persistent workers: same base seed, but next epoch
        assert first != third  # new base seed
        # not in natural order
        assert first[0][0] != list(range(7))

    def test_data_loader_multiple_workers(self):
        """Test iteration with a real multi-process data loader with persistent workers."""
        # 52 triples with batch size 8 -> 7 batches; a per-worker split (26 triples each) would yield 2 x 4 = 8 batches
        num_triples, batch_size = 52, 8
        mapped_triples = self.factory.mapped_triples[:num_triples]
        # map (unique) triples back to their IDs
        triple_to_id = {tuple(triple): i for i, triple in enumerate(mapped_triples.tolist())}
        assert len(triple_to_id) == num_triples
        instance = BatchedSLCWAInstances(
            mapped_triples=mapped_triples,
            batch_size=batch_size,
            drop_last=False,
            num_entities=self.factory.num_entities,
            num_relations=self.factory.num_relations,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=instance,
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(42),
        )
        epochs = []
        for _ in range(2):
            batches = [[triple_to_id[tuple(triple)] for triple in batch["positives"].tolist()] for batch in data_loader]
            assert len(batches) == len(instance)
            assert sorted(i for batch in batches for i in batch) == list(range(num_triples))
            epochs.append(batches)
        assert epochs[0] != epochs[1]

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

    def test_grouped_with_loss_weighter(self):
        """Test that grouped instances with a loss weighter emit correctly-shaped neg_weights.

        Regression test: the per-target loss weighter call relies on broadcasting semantics (cf.
        :meth:`pykeen.triples.weights.LossWeighter.__call__`), e.g. :class:`~pykeen.triples.weights.
        RelationLossWeighter` ignores h/t and returns a `(batch, 1)`-shaped tensor regardless of how many
        corruptions were requested per target -- this must be broadcast to match the corruptions' shape before
        being consumed downstream (cf. :meth:`pykeen.training.slcwa.SLCWATrainingLoop._process_grouped_batch_static`).
        """
        num_negs_per_pos = 3
        weighter = RelationLossWeighter.inverse_relation_frequency(self.factory.mapped_triples)
        instances = BatchedSLCWAInstances.from_triples_factory(
            self.factory,
            batch_size=2,
            negative_sampler_kwargs={"num_negs_per_pos": num_negs_per_pos},
            loss_weighter=weighter,
            grouped=True,
        )
        for batch in torch.utils.data.DataLoader(dataset=instances, batch_size=None):
            assert batch["pos_weights"].shape == (2,)
            for target, replacements in batch["corruptions"].items():
                assert batch["neg_weights"][target].shape == replacements.shape

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
