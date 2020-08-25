# -*- coding: utf-8 -*-

"""Schlichtkrull Sampler Class."""

import logging
from typing import Optional, Tuple

import torch
from torch.utils.data.sampler import Sampler

from ..triples import TriplesFactory


def _compute_compressed_adjacency_list(
    triples_factory: TriplesFactory,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Compute compressed undirected adjacency list representation for efficient sampling.

    The compressed adjacency list format is inspired by CSR sparse matrix format.

    :param triples_factory: The triples factory.
    :return: a tuple (degrees, offsets, compressed_adj_lists)
        where
            degrees: shape: (num_entities,)
            offsets: shape: (num_entities,)
            compressed_adj_list: shape: (2*num_triples, 2)
        with
            adj_list[i] = compressed_adj_list[offsets[i]:offsets[i+1]]
    """
    adj_lists = [[] for _ in range(triples_factory.num_entities)]
    for i, (s, _, o) in enumerate(triples_factory.mapped_triples):
        adj_lists[s].append([i, o.item()])
        adj_lists[o].append([i, s.item()])
    degrees = torch.tensor([len(a) for a in adj_lists], dtype=torch.long)
    assert torch.sum(degrees) == 2 * triples_factory.num_triples

    offset = torch.empty(triples_factory.num_entities, dtype=torch.long)
    offset[0] = 0
    offset[1:] = torch.cumsum(degrees, dim=0)[:-1]
    compressed_adj_lists = torch.cat([torch.as_tensor(adj_list, dtype=torch.long) for adj_list in adj_lists], dim=0)
    return degrees, offset, compressed_adj_lists


class GraphSampler(Sampler):
    r"""Samples edges based on the proposed method in Schlichtkrull et al.

    .. seealso::

        https://github.com/MichSchli/RelationPrediction/blob/2560e4ea7ccae5cb4f877ac7cb1dc3924f553827/code/train.py#L161-L247
    """

    def __init__(
        self,
        triples_factory: TriplesFactory,
        num_samples: Optional[int] = None,
    ):
        mapped_triples = triples_factory.mapped_triples
        super().__init__(data_source=mapped_triples)
        self.triples_factory = triples_factory

        if num_samples is None:
            num_samples = triples_factory.num_triples // 10
            logging.info(f'Did not specify number of samples. Using {num_samples}.')
        elif num_samples > triples_factory.num_triples:
            raise ValueError(
                'num_samples cannot be larger than the number of triples, but '
                f'{num_samples} > {triples_factory.num_triples}.',
            )
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")

        self.num_samples = num_samples
        self.num_batches_per_epoch = triples_factory.num_triples // self.num_samples

        # preprocessing
        self.degrees, self.offset, self.neighbors = _compute_compressed_adjacency_list(triples_factory=triples_factory)

    def __iter__(self):  # noqa: D105
        # initialize
        chosen_edges = torch.empty(self.num_samples, dtype=torch.long)
        node_weights = self.degrees.detach().clone()
        edge_picked = torch.zeros(self.triples_factory.num_triples, dtype=torch.bool)
        node_picked = torch.zeros(self.triples_factory.num_entities, dtype=torch.bool)

        # sample iteratively
        for i in range(0, self.num_samples):
            # determine weights
            weights = node_weights * node_picked

            # only happens at first iteration
            if torch.sum(weights) == 0:
                weights = torch.ones_like(weights)
                weights[node_weights == 0] = 0
                assert i == 0
            else:
                assert i > 0

            # normalize to probabilities
            probabilities = weights.float() / weights.sum().float()

            # sample a start node
            chosen_vertex = torch.multinomial(probabilities, num_samples=1)[0]
            node_picked[chosen_vertex] = True

            # get list of neighbors
            start = self.offset[chosen_vertex]
            chosen_node_degree = self.degrees[chosen_vertex]
            stop = start + chosen_node_degree
            adj_list = self.neighbors[start:stop, :]

            # sample an outgoing edge at random which has not been chosen yet using rejection sampling
            chosen_edge_index = torch.randint(chosen_node_degree, size=(1,))[0]
            chosen_edge = adj_list[chosen_edge_index]
            edge_number = chosen_edge[0]
            while edge_picked[edge_number]:
                chosen_edge_index = torch.randint(chosen_node_degree, size=(1,))[0]
                chosen_edge = adj_list[chosen_edge_index]
                edge_number = chosen_edge[0]
            chosen_edges[i] = edge_number
            edge_picked[edge_number] = True

            # visit target node
            other_vertex = chosen_edge[1]
            node_picked[other_vertex] = True

            # decrease sample counts
            node_weights[chosen_vertex] -= 1
            node_weights[other_vertex] -= 1

        # return chosen edges
        return iter(chosen_edges)

    def __len__(self):  # noqa: D105
        return self.num_batches_per_epoch
