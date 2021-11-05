# -*- coding: utf-8 -*-

"""Schlichtkrull Sampler Class."""

import logging
from typing import Iterator, List, Optional, Tuple
from abc import ABC
from typing import List, Optional, Tuple

import torch
from torch.utils.data.sampler import Sampler

from ..triples.instances import BatchType, Instances, SLCWABatchType, SLCWAInstances
from ..typing import MappedTriples


def _compute_compressed_adjacency_list(
    mapped_triples: MappedTriples,
    num_entities: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Compute compressed undirected adjacency list representation for efficient sampling.

    The compressed adjacency list format is inspired by CSR sparse matrix format.

    :param mapped_triples:
        the ID-based triples
    :param num_entities:
        the number of entities.

    :return: a tuple (degrees, offsets, compressed_adj_lists)
        where
            degrees: shape: (num_entities,)
            offsets: shape: (num_entities,)
            compressed_adj_list: shape: (2*num_triples, 2)
        with
            adj_list[i] = compressed_adj_list[offsets[i]:offsets[i+1]]
    """
    num_entities = num_entities or mapped_triples[:, [0, 2]].max().item() + 1
    num_triples = mapped_triples.shape[0]
    adj_lists: List[List[Tuple[int, float]]] = [[] for _ in range(num_entities)]
    for i, (s, _, o) in enumerate(mapped_triples):
        adj_lists[s].append((i, o.item()))
        adj_lists[o].append((i, s.item()))
    degrees = torch.tensor([len(a) for a in adj_lists], dtype=torch.long)
    assert torch.sum(degrees) == 2 * num_triples

    offset = torch.empty(num_entities, dtype=torch.long)
    offset[0] = 0
    offset[1:] = torch.cumsum(degrees, dim=0)[:-1]
    compressed_adj_lists = torch.cat([torch.as_tensor(adj_list, dtype=torch.long) for adj_list in adj_lists], dim=0)
    return degrees, offset, compressed_adj_lists


class GraphSampler(Sampler):
    r"""Samples edges based on the proposed method in Schlichtkrull et al.

    .. seealso::

        https://github.com/MichSchli/RelationPrediction/blob/2560e4ea7ccae5cb4f877ac7cb1dc3924f553827/code/train.py#L161-L247

    To be used as a *batch* sampler.
    """

    def __init__(
        self,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
    ):
        super().__init__(data_source=mapped_triples)
        self.num_triples = num_triples = mapped_triples.shape[0]

        if batch_size is None:
            batch_size = num_triples // 10
            logging.info(f"Did not specify number of samples. Using {batch_size}.")
        elif batch_size > num_triples:
            raise ValueError(
                "num_samples cannot be larger than the number of triples, but " f"{batch_size} > {num_triples}.",
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={batch_size}")

        self.num_samples = batch_size
        self.num_batches_per_epoch = num_triples // self.num_samples

        # preprocessing
        self.degrees, self.offset, self.neighbors = _compute_compressed_adjacency_list(mapped_triples=mapped_triples)

    def _sample_batch(self) -> List[int]:
        """Sample one batch."""
        # initialize
        chosen_edges = torch.empty(self.num_samples, dtype=torch.long)
        node_weights = self.degrees.detach().clone()
        edge_picked = torch.zeros(self.num_triples, dtype=torch.bool)
        node_picked = torch.zeros(self.degrees.shape[0], dtype=torch.bool)

        # sample iteratively
        for i in range(self.num_samples):
            # determine weights
            weights = node_weights * node_picked

            if torch.sum(weights) == 0:
                # randomly choose a vertex which has not been chosen yet
                pool = (~node_picked).nonzero()
                chosen_vertex = pool[torch.randint(pool.numel(), size=tuple())]
            else:
                # normalize to probabilities
                probabilities = weights.float() / weights.sum().float()
                chosen_vertex = torch.multinomial(probabilities, num_samples=1)[0]

            # sample a start node
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
        return chosen_edges

    def __iter__(self) -> Iterator[List[int]]:  # noqa: D105
        for _ in range(self.num_batches_per_epoch):
            yield self._sample_batch()

    def __len__(self):  # noqa: D105
        return self.num_batches_per_epoch


class SubGraphInstances(Instances[BatchType], ABC):
    """A dataset of subgraph samples."""

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        sub_graph_size: int,
        **kwargs,
    ) -> None:
        """
        Initialize the subgraph dataset.

        :param mapped_triples:
            the ID-based triples
        :param sub_graph_size:
            the size (=number of triples) of the individual subgraph samples
        :param kwargs:
            additional keyword based arguments passed to super.__init__
        """
        super().__init__(**kwargs)
        self.graph_sampler = GraphSampler(
            mapped_triples=mapped_triples,
            num_samples=sub_graph_size,
        )


class SLCWASubGraphInstances(SLCWAInstances, SubGraphInstances[SLCWABatchType]):
    """SLCWA subgraph instances."""

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        sub_graph_size: int,
    ):
        """
        Initialize the subgraph dataset for SLCWA instances.

        :param mapped_triples:
            the ID-based triples.
        :param sub_graph_size:
            the size (=number of triples) of the individual subgraph samples
        """
        SLCWAInstances.__init__(self, mapped_triples=mapped_triples)
        SubGraphInstances.__init__(self, mapped_triples=mapped_triples, sub_graph_size=sub_graph_size)

    def __len__(self) -> int:  # noqa: D105
        # is already batched!
        return super().__len__() // self.graph_sampler.num_samples

    def __getitem__(self, item: int) -> MappedTriples:  # noqa: D105
        return torch.stack([SLCWAInstances.__getitem__(self, idx) for idx in self.graph_sampler], dim=0)
