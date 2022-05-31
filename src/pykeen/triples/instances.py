# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import math
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Iterator, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import scipy.sparse
import torch
from class_resolver import HintOrType, OptionalKwargs
from torch.utils import data

from .utils import compute_compressed_adjacency_list
from ..sampling import NegativeSampler, negative_sampler_resolver
from ..typing import MappedTriples

__all__ = [
    "Instances",
    "SLCWAInstances",
    "LCWAInstances",
]

# TODO: the same
SampleType = TypeVar("SampleType")
BatchType = TypeVar("BatchType")
LCWASampleType = Tuple[MappedTriples, torch.FloatTensor]
LCWABatchType = Tuple[MappedTriples, torch.FloatTensor]
SLCWASampleType = Tuple[MappedTriples, MappedTriples, Optional[torch.BoolTensor]]


class SLCWABatch(NamedTuple):
    """A batch for sLCWA training."""

    #: the positive triples, shape: (batch_size, 3)
    positives: torch.LongTensor

    #: the negative triples, shape: (batch_size, num_negatives_per_positive, 3)
    negatives: torch.LongTensor

    #: filtering masks for negative triples, shape: (batch_size, num_negatives_per_positive)
    masks: Optional[torch.BoolTensor]


class Instances(data.Dataset[BatchType], Generic[SampleType, BatchType], ABC):
    """Base class for training instances."""

    def __len__(self):  # noqa:D401
        """The number of instances."""
        raise NotImplementedError

    def get_collator(self) -> Optional[Callable[[List[SampleType]], BatchType]]:
        """Get a collator."""
        return None

    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        **kwargs,
    ) -> "Instances":
        """Create instances from mapped triples.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        :param num_entities: >0
            The number of entities.
        :param num_relations: >0
            The number of relations.
        :param kwargs:
            additional keyword-based parameters.

        :return:
            The instances.
        """
        raise NotImplementedError


class SLCWAInstances(Instances[SLCWASampleType, SLCWABatch]):
    """Training instances for the sLCWA."""

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
    ):
        """Initialize the sLCWA instances.

        :param mapped_triples: shape: (num_triples, 3)
            the ID-based triples, passed to the negative sampler
        :param num_entities: >0
            the number of entities, passed to the negative sampler
        :param num_relations: >0
            the number of relations, passed to the negative sampler
        :param negative_sampler:
            the negative sampler, or a hint thereof
        :param negative_sampler_kwargs:
            additional keyword-based arguments passed to the negative sampler
        """
        self.mapped_triples = mapped_triples
        self.sampler = negative_sampler_resolver.make(
            negative_sampler,
            pos_kwargs=negative_sampler_kwargs,
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
        )

    def __len__(self) -> int:  # noqa: D105
        return self.mapped_triples.shape[0]

    def __getitem__(self, item: int) -> SLCWASampleType:  # noqa: D105
        positive = self.mapped_triples[item].unsqueeze(dim=0)
        # TODO: some negative samplers require batches
        negative, mask = self.sampler.sample(positive_batch=positive)
        # shape: (1, 3), (1, k, 3), (1, k, 3)?
        return positive, negative, mask

    @staticmethod
    def collate(samples: Iterable[SLCWASampleType]) -> SLCWABatch:
        """Collate samples."""
        # each shape: (1, 3), (1, k, 3), (1, k, 3)?
        positives, negatives, masks = zip(*samples)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        if masks[0] is None:
            assert all(m is None for m in masks)
            masks = None
        else:
            masks = torch.cat(masks, dim=0)
        return SLCWABatch(positives, negatives, masks)

    # docstr-coverage: inherited
    def get_collator(self) -> Optional[Callable[[List[SLCWASampleType]], SLCWABatch]]:  # noqa: D102
        return self.collate

    # docstr-coverage: inherited
    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        **kwargs,
    ) -> Instances:  # noqa: D102
        return cls(mapped_triples=mapped_triples, num_entities=num_entities, num_relations=num_relations, **kwargs)


class BaseBatchedSLCWAInstances(data.IterableDataset[SLCWABatch]):
    """
    Pre-batched training instances for the sLCWA training loop.

    .. note ::
        this class is intended to be used with automatic batching disabled, i.e., both parameters `batch_size` and
        `batch_sampler` of torch.utils.data.DataLoader` are set to `None`.
    """

    def __init__(
        self,
        mapped_triples: MappedTriples,
        batch_size: int = 1,
        drop_last: bool = True,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
    ):
        """
        Initialize the dataset.

        :param mapped_triples: shape: (num_triples, 3)
            the mapped triples
        :param batch_size:
            the batch size
        :param drop_last:
            whether to drop the last (incomplete) batch
        :param num_entities: >0
            the number of entities, passed to the negative sampler
        :param num_relations: >0
            the number of relations, passed to the negative sampler
        :param negative_sampler:
            the negative sampler, or a hint thereof
        :param negative_sampler_kwargs:
            additional keyword-based parameters used to instantiate the negative sampler
        """
        self.mapped_triples = mapped_triples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.negative_sampler = negative_sampler_resolver.make(
            negative_sampler,
            pos_kwargs=negative_sampler_kwargs,
            mapped_triples=self.mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
        )

    def __getitem__(self, item: List[int]) -> SLCWABatch:
        """Get a batch from the given list of positive triple IDs."""
        positive_batch = self.mapped_triples[item]
        negative_batch, masks = self.negative_sampler.sample(positive_batch=positive_batch)
        return SLCWABatch(positives=positive_batch, negatives=negative_batch, masks=masks)

    def split_workload(self, n: int) -> range:
        """Split workload for multi-processing."""
        # cf. https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            workload = range(n)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id  # 1-based
            start = math.ceil(n / num_workers * worker_id)
            stop = math.ceil(n / num_workers * (worker_id + 1))
            workload = range(start, stop)
        return workload

    @abstractmethod
    def iter_triple_ids(self) -> Iterable[List[int]]:
        """Iterate over batches of IDs of positive triples."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[SLCWABatch]:
        """Iterate over batches."""
        for triple_ids in self.iter_triple_ids():
            yield self[triple_ids]

    def __len__(self) -> int:
        """Return the number of batches."""
        num_batches, remainder = divmod(len(self.mapped_triples), self.batch_size)
        if remainder and not self.drop_last:
            num_batches += 1
        return num_batches


class BatchedSLCWAInstances(BaseBatchedSLCWAInstances):
    """Random pre-batched training instances for the sLCWA training loop."""

    # docstr-coverage: inherited
    def iter_triple_ids(self) -> Iterable[List[int]]:  # noqa: D102
        yield from data.BatchSampler(
            sampler=data.RandomSampler(data_source=self.split_workload(len(self.mapped_triples))),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )


class SubGraphSLCWAInstances(BaseBatchedSLCWAInstances):
    """Pre-batched training instances for SLCWA of coherent subgraphs."""

    def __init__(self, **kwargs):
        """
        Initialize the instances.

        :param kwargs:
            keyword-based parameters passed to :meth:`BaseBatchedSLCWAInstances.__init__`
        """
        super().__init__(**kwargs)
        # indexing
        self.degrees, self.offset, self.neighbors = compute_compressed_adjacency_list(
            mapped_triples=self.mapped_triples
        )

    def subgraph_sample(self) -> List[int]:
        """Sample one subgraph."""
        # initialize
        node_weights = self.degrees.detach().clone()
        edge_picked = torch.zeros(self.mapped_triples.shape[0], dtype=torch.bool)
        node_picked = torch.zeros(self.degrees.shape[0], dtype=torch.bool)

        # sample iteratively
        result = []
        for _ in range(self.batch_size):
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
            chosen_node_degree = self.degrees[chosen_vertex].item()
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
            result.append(edge_number.item())

            edge_picked[edge_number] = True

            # visit target node
            other_vertex = chosen_edge[1]
            node_picked[other_vertex] = True

            # decrease sample counts
            node_weights[chosen_vertex] -= 1
            node_weights[other_vertex] -= 1
        return result

    # docstr-coverage: inherited
    def iter_triple_ids(self) -> Iterable[List[int]]:  # noqa: D102
        yield from (self.subgraph_sample() for _ in self.split_workload(n=len(self)))


class LCWAInstances(Instances[LCWASampleType, LCWABatchType]):
    """Triples and mappings to their indices for LCWA."""

    def __init__(self, *, pairs: np.ndarray, compressed: scipy.sparse.csr_matrix):
        """Initialize the LCWA instances.

        :param pairs: The unique pairs
        :param compressed: The compressed triples in CSR format
        """
        self.pairs = pairs
        self.compressed = compressed

    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        target: Optional[int] = None,
        **kwargs,
    ) -> Instances:
        """
        Create LCWA instances from triples.

        :param mapped_triples: shape: (num_triples, 3)
            The ID-based triples.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param target:
            The column to predict

        :return:
            The instances.
        """
        if target is None:
            target = 2
        mapped_triples = mapped_triples.numpy()
        other_columns = sorted(set(range(3)).difference({target}))
        unique_pairs, pair_idx_to_triple_idx = np.unique(mapped_triples[:, other_columns], return_inverse=True, axis=0)
        num_pairs = unique_pairs.shape[0]
        tails = mapped_triples[:, target]
        target_size = num_relations if target == 1 else num_entities
        compressed = scipy.sparse.coo_matrix(
            (np.ones(mapped_triples.shape[0], dtype=np.float32), (pair_idx_to_triple_idx, tails)),
            shape=(num_pairs, target_size),
        )
        # convert to csr for fast row slicing
        compressed = compressed.tocsr()
        return cls(pairs=unique_pairs, compressed=compressed)

    def __len__(self) -> int:  # noqa: D105
        return self.pairs.shape[0]

    def __getitem__(self, item: int) -> LCWABatchType:  # noqa: D105
        return self.pairs[item], np.asarray(self.compressed[item, :].todense())[0, :]
