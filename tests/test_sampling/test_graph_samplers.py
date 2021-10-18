# -*- coding: utf-8 -*-

"""Tests for graph samplers."""

import unittest

import torch

from pykeen.datasets import Nations
from pykeen.training.schlichtkrull_sampler import GraphSampler, _compute_compressed_adjacency_list


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
                batch_indices.append(torch.as_tensor(j))
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
            adjacent_edges = set(
                int(a) for a in ((triples[:, 0] == i) | (triples[:, 2] == i)).nonzero(as_tuple=False).flatten()
            )
            assert adjacent_edges == set(map(int, edge_ids))
