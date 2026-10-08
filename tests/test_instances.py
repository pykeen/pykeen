"""Tests for training instances."""

import copy
import itertools
from collections import Counter
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
        factory = TriplesFactory.from_labeled_triples(triples=t)
        instances = BatchedSLCWAInstances.from_triples_factory(factory, create_inverse_triples=True)
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

    def test_iter_triple_ids_single_process(self):
        """Test the number and sizes of batches, and coverage, for single-process loading."""
        for num_triples, batch_size, drop_last in itertools.product((13, 14, 50), (1, 3, 7), (False, True)):
            with self.subTest(num_triples=num_triples, batch_size=batch_size, drop_last=drop_last):
                instance = BatchedSLCWAInstances(
                    mapped_triples=self.factory.mapped_triples[:num_triples],
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
                batches = list(instance.iter_triple_ids())
                assert len(batches) == len(instance)
                num_full_batches, remainder = divmod(num_triples, batch_size)
                expected_sizes = [batch_size] * num_full_batches
                if remainder and not drop_last:
                    expected_sizes.append(remainder)
                assert [len(batch) for batch in batches] == expected_sizes
                triple_ids = [i for batch in batches for i in batch]
                assert len(triple_ids) == len(set(triple_ids))
                assert set(triple_ids).issubset(range(num_triples))

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

    def test_subgraph_sample(self):
        """Test that a subgraph sample consists of batch_size unique, valid triple IDs."""
        instance = SubGraphSLCWAInstances(mapped_triples=self.factory.mapped_triples, batch_size=64)
        for _ in range(10):
            triple_ids = instance.subgraph_sample()
            assert len(triple_ids) == 64
            assert len(set(triple_ids)) == 64
            assert all(isinstance(i, int) and 0 <= i < self.factory.num_triples for i in triple_ids)

    def test_subgraph_sample_reproducible(self):
        """Test that subgraph sampling is reproducible via torch's global seed."""
        torch.manual_seed(42)
        first = [self.instance.subgraph_sample() for _ in range(3)]
        torch.manual_seed(42)
        second = [self.instance.subgraph_sample() for _ in range(3)]
        assert first == second

    def test_subgraph_sample_all(self):
        """Test sampling all triples of a graph with isolated vertices, self-loops, and duplicate edges."""
        mapped_triples = torch.as_tensor(
            # vertex 3 is isolated
            [[0, 0, 1], [1, 0, 2], [2, 0, 2], [0, 0, 1], [4, 0, 5], [5, 1, 5], [6, 0, 4]],
            dtype=torch.long,
        )
        num_triples = mapped_triples.shape[0]
        instance = SubGraphSLCWAInstances(mapped_triples=mapped_triples, batch_size=num_triples)
        for _ in range(20):
            assert sorted(instance.subgraph_sample()) == list(range(num_triples))
        # requesting more triples than there are returns all of them
        for size in (num_triples + 1, 2 * num_triples):
            assert sorted(instance.subgraph_sample(size=size)) == list(range(num_triples))
        instance.batch_size = num_triples + 1
        assert sorted(instance.subgraph_sample()) == list(range(num_triples))

    def test_subgraph_sample_size(self):
        """Test sampling subgraphs of an explicitly requested size."""
        assert self.instance.batch_size > 1
        for size in range(self.instance.batch_size + 2):
            triple_ids = self.instance.subgraph_sample(size=size)
            assert len(triple_ids) == size
            assert len(set(triple_ids)) == size
            assert all(0 <= i < self.factory.num_triples for i in triple_ids)
        with pytest.raises(ValueError, match="non-negative"):
            self.instance.subgraph_sample(size=-1)

    def test_subgraph_sample_distribution(self):
        """Test that the empirical distribution of subgraph samples matches the exact one."""
        mapped_triples = torch.as_tensor([[0, 0, 1], [1, 0, 2], [1, 0, 3], [3, 0, 3], [4, 0, 5]], dtype=torch.long)
        batch_size = 3
        expected = _subgraph_sample_distribution(edges=mapped_triples[:, [0, 2]].tolist(), batch_size=batch_size)
        instance = SubGraphSLCWAInstances(mapped_triples=mapped_triples, batch_size=batch_size)
        torch.manual_seed(0)
        num_samples = 20_000
        counts = Counter(tuple(instance.subgraph_sample()) for _ in range(num_samples))
        assert set(counts).issubset(expected)
        total_variation = 0.5 * sum(abs(counts[key] / num_samples - p) for key, p in expected.items())
        assert total_variation < 0.02

    def test_iter_triple_ids_batch_sizes(self):
        """Test the number and sizes of batches for single- and (simulated) multi-process loading."""
        # the first 13 triples leave some entities without incident edges; for these, batch size 4 leaves a remainder
        # of 1, 13 divides evenly, and 20 exceeds the number of triples. All 1592 triples: 398 divides evenly, 300
        # leaves a remainder of 92.
        sizes = [(13, 4), (13, 13), (13, 20), (1592, 300), (1592, 398)]
        assert self.factory.num_triples == 1592
        for (num_triples, batch_size), num_workers, drop_last in itertools.product(
            sizes, (None, 1, 2, 3), (False, True)
        ):
            with self.subTest(
                num_triples=num_triples, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last
            ):
                instance = SubGraphSLCWAInstances(
                    mapped_triples=self.factory.mapped_triples[:num_triples], batch_size=batch_size, drop_last=drop_last
                )
                if num_triples == 13:
                    assert (instance.degrees == 0).any()
                if num_workers is None:
                    batches = list(instance.iter_triple_ids())
                else:
                    batches = []
                    for worker_id in range(num_workers):
                        worker_info = mock.Mock(id=worker_id, num_workers=num_workers)
                        with mock.patch("torch.utils.data.get_worker_info", return_value=worker_info):
                            batches.extend(copy.deepcopy(instance).iter_triple_ids())
                assert len(batches) == len(instance)
                num_full_batches, remainder = divmod(num_triples, batch_size)
                expected_sizes = [batch_size] * num_full_batches
                if remainder and not drop_last:
                    expected_sizes.append(remainder)
                assert [len(batch) for batch in batches] == expected_sizes
                for batch in batches:
                    # each subgraph consists of distinct, valid triples
                    assert len(set(batch)) == len(batch)
                    assert set(batch).issubset(range(num_triples))

    def test_data_loader_multiple_workers(self):
        """Test that the total number of sampled triples with a real multi-process data loader matches."""
        instance = SubGraphSLCWAInstances(
            mapped_triples=self.factory.mapped_triples,
            batch_size=300,
            drop_last=False,
            num_entities=self.factory.num_entities,
            num_relations=self.factory.num_relations,
        )
        sizes = [
            batch["positives"].shape[0]
            for batch in torch.utils.data.DataLoader(dataset=instance, batch_size=None, num_workers=2)
        ]
        assert len(sizes) == len(instance)
        assert sum(sizes) == self.factory.num_triples
        assert sorted(sizes) == [92] + [300] * 5


def _subgraph_sample_distribution(edges: list[list[int]], batch_size: int) -> dict[tuple[int, ...], float]:
    """Compute the exact distribution over ordered subgraph samples by enumerating the sampling process."""
    num_vertices = max(itertools.chain.from_iterable(edges)) + 1
    # half-edges per vertex; a self-loop occurs twice in its vertex's adjacency list
    half_edges: list[list[tuple[int, int]]] = [[] for _ in range(num_vertices)]
    for i, (s, o) in enumerate(edges):
        half_edges[s].append((i, o))
        half_edges[o].append((i, s))
    result: dict[tuple[int, ...], float] = {}

    def _pick(vertex: int, prob: float, visited: frozenset[int], picked: tuple[int, ...]) -> None:
        # choose uniformly among the vertex' adjacency list entries whose edge has not been picked yet
        candidates = [(i, o) for i, o in half_edges[vertex] if i not in picked]
        for i, o in candidates:
            _step(prob / len(candidates), visited | {o}, (*picked, i))

    def _step(prob: float, visited: frozenset[int], picked: tuple[int, ...]) -> None:
        if len(picked) == batch_size:
            result[picked] = result.get(picked, 0.0) + prob
            return
        weights = {v: sum(i not in picked for i, _ in half_edges[v]) for v in visited}
        total = sum(weights.values())
        if total:
            # degree-weighted choice among visited vertices
            for vertex, weight in weights.items():
                if weight:
                    _pick(vertex, prob * weight / total, visited, picked)
        else:
            # uniform choice among unvisited (non-isolated) vertices
            pool = [v for v in range(num_vertices) if v not in visited and half_edges[v]]
            for vertex in pool:
                _pick(vertex, prob / len(pool), visited | {vertex}, picked)

    _step(1.0, frozenset(), ())
    return result
