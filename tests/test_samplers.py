# -*- coding: utf-8 -*-

"""Test that samplers can be executed."""

import unittest
from typing import ClassVar, Type

import numpy
import torch

from pykeen.datasets import Nations
from pykeen.sampling import BasicNegativeSampler, BernoulliNegativeSampler, NegativeSampler
from pykeen.training.schlichtkrull_sampler import GraphSampler, _compute_compressed_adjacency_list
from pykeen.triples import SLCWAInstances, TriplesFactory


def _array_check_bounds(
    array: torch.LongTensor,
    low: int,
    high: int,
) -> bool:
    """Check if all elements lie in bounds."""
    return (low <= array).all() and (array < high).all()


class _NegativeSamplingTestCase:
    """A test case for quickly defining common tests for samplers."""

    #: The batch size
    batch_size: int
    #: The random seed
    seed: int
    #: The triples factory
    triples_factory: TriplesFactory
    #: The sLCWA instances
    slcwa_instances: SLCWAInstances
    #: Class of negative sampling to test
    negative_sampling_cls: ClassVar[Type[NegativeSampler]]
    #: The negative sampler instance, initialized in setUp
    negative_sampler: NegativeSampler
    #: A positive batch
    positive_batch: torch.LongTensor

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.batch_size = 16
        self.seed = 42
        self.num_negs_per_pos = 10
        self.triples_factory = Nations().training
        self.slcwa_instances = self.triples_factory.create_slcwa_instances()
        self.negative_sampler = self.negative_sampling_cls(triples_factory=self.triples_factory)
        self.scaling_negative_sampler = self.negative_sampling_cls(
            triples_factory=self.triples_factory,
            num_negs_per_pos=self.num_negs_per_pos,
        )
        random = numpy.random.RandomState(seed=self.seed)
        batch_indices = random.randint(low=0, high=self.slcwa_instances.num_instances, size=(self.batch_size,))
        self.positive_batch = self.slcwa_instances.mapped_triples[batch_indices]

    def test_sample(self) -> None:
        # Generate negative sample
        negative_batch = self.negative_sampler.sample(positive_batch=self.positive_batch)

        # check shape
        assert negative_batch.shape == self.positive_batch.shape

        # check bounds: heads
        assert _array_check_bounds(negative_batch[:, 0], low=0, high=self.triples_factory.num_entities)

        # check bounds: relations
        assert _array_check_bounds(negative_batch[:, 1], low=0, high=self.triples_factory.num_relations)

        # check bounds: tails
        assert _array_check_bounds(negative_batch[:, 2], low=0, high=self.triples_factory.num_entities)

        # Check that all elements got corrupted
        assert (negative_batch != self.positive_batch).any(dim=1).all()

        # Generate scaled negative sample
        scaled_negative_batch = self.scaling_negative_sampler.sample(
            positive_batch=self.positive_batch,
        )

        assert scaled_negative_batch.shape[0] == self.positive_batch.shape[0] * self.num_negs_per_pos
        assert scaled_negative_batch.shape[1] == self.positive_batch.shape[1]


class BasicNegativeSamplerTest(_NegativeSamplingTestCase, unittest.TestCase):
    """Test the basic negative sampler."""

    negative_sampling_cls = BasicNegativeSampler

    def test_sample_basic(self):
        """Test if relations and half of heads and tails are not corrupted."""
        # Generate negative samples
        negative_batch = self.negative_sampler.sample(positive_batch=self.positive_batch)

        # test that the relations were not changed
        assert (self.positive_batch[:, 1] == negative_batch[:, 1]).all()

        # Test that half of the subjects and half of the objects are corrupted
        half_size = self.positive_batch.shape[0] // 2
        num_subj_corrupted = (self.positive_batch[:, 0] != negative_batch[:, 0]).sum()
        num_obj_corrupted = (self.positive_batch[:, 2] != negative_batch[:, 2]).sum()
        assert num_obj_corrupted - 1 <= num_subj_corrupted
        assert num_subj_corrupted - 1 <= num_obj_corrupted
        assert num_subj_corrupted - 1 <= self.positive_batch.shape[0]
        assert half_size - 1 <= num_subj_corrupted


class BernoulliNegativeSamplerTest(_NegativeSamplingTestCase, unittest.TestCase):
    """Test the Bernoulli negative sampler."""

    negative_sampling_cls = BernoulliNegativeSampler

    def test_sample_bern(self):
        """Test if relations are not corrupted."""
        # Generate negative sample for additional tests
        negative_batch = self.negative_sampler.sample(positive_batch=self.positive_batch)

        # test that the relations were not changed
        assert (self.positive_batch[:, 1] == negative_batch[:, 1]).all()


class GraphSamplerTest(unittest.TestCase):
    """Test the GraphSampler."""

    def setUp(self) -> None:
        """Set up the test case with a triples factory."""
        self.triples_factory = Nations().training
        self.num_samples = 20
        self.num_epochs = 10
        self.graph_sampler = GraphSampler(triples_factory=self.triples_factory, num_samples=self.num_samples)

    def test_sample(self) -> None:
        """Test drawing samples from GraphSampler."""
        for e in range(self.num_epochs):
            # sample a batch
            batch_indices = []
            for j in self.graph_sampler:
                batch_indices.append(j)
            batch = torch.stack(batch_indices)

            # check shape
            assert batch.shape == (self.num_samples,)

            # get triples
            triples_batch = self.triples_factory.mapped_triples[batch]

            # check connected components
            # super inefficient
            components = [{int(e)} for e in torch.cat([triples_batch[:, i] for i in (0, 2)]).unique()]
            for h, _, t in triples_batch:
                h, t = int(h), int(t)

                s_comp_ind = [i for i, c in enumerate(components) if h in c][0]
                o_comp_ind = [i for i, c in enumerate(components) if t in c][0]

                # join
                if s_comp_ind != o_comp_ind:
                    s_comp = components.pop(max(s_comp_ind, o_comp_ind))
                    o_comp = components.pop(min(s_comp_ind, o_comp_ind))
                    so_comp = s_comp.union(o_comp)
                    components.append(so_comp)
                else:
                    pass
                    # already joined

                if len(components) < 2:
                    break

            # check that there is only a single component
            assert len(components) == 1


class AdjacencyListCompressionTest(unittest.TestCase):
    """Unittest for utility method."""

    def setUp(self) -> None:
        """Set up the test case with a triples factory."""
        self.triples_factory = Nations().training

    def test_compute_compressed_adjacency_list(self):
        """Test method _compute_compressed_adjacency_list ."""
        degrees, offsets, comp_adj_lists = _compute_compressed_adjacency_list(triples_factory=self.triples_factory)
        triples = self.triples_factory.mapped_triples
        uniq, cnt = torch.unique(torch.cat([triples[:, i] for i in (0, 2)]), return_counts=True)
        assert (degrees == cnt).all()
        assert (offsets[1:] == torch.cumsum(cnt, dim=0)[:-1]).all()
        assert (offsets < comp_adj_lists.shape[0]).all()

        # check content of comp_adj_lists
        for i in range(self.triples_factory.num_entities):
            start = offsets[i]
            stop = start + degrees[i]
            adj_list = comp_adj_lists[start:stop]

            # check edge ids
            edge_ids = adj_list[:, 0]
            adjacent_edges = set(int(a) for a in ((triples[:, 0] == i) | (triples[:, 2] == i)).nonzero().flatten())
            assert adjacent_edges == set(map(int, edge_ids))
