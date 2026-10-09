"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Generic, NotRequired, Self, TypedDict, TypeVar, Any

import numpy as np
import scipy.sparse
import torch
from class_resolver import HintOrType, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from torch.utils import data

from .triples_factory import CoreTriplesFactory
from .utils import compute_compressed_adjacency_list
from .weights import LossWeighter, loss_weighter_resolver
from .. import typing as pykeen_typing
from ..constants import get_target_column
from ..sampling import BasicNegativeSampler, NegativeSampler, negative_sampler_resolver
from ..typing import (
    BoolTensor,
    FloatTensor,
    LongTensor,
    MappedTriples,
    Target,
    TargetColumn,
    TargetHint,
)
from ..utils import split_workload

__all__ = [
    "BaseBatchedSLCWAInstances",
    "BatchCWABatch",
    "BatchedSLCWAInstances",
    "GroupedSLCWABatch",
    "Instances",
    "LCWABatch",
    "LCWAInstances",
    "SLCWABatch",
    "SubGraphSLCWAInstances",
]

BatchType = TypeVar("BatchType")


class BatchCWABatch(TypedDict):
    """A batch for BCWA training."""

    heads: LongTensor
    """The unique head entity indices, shape: (num_unique_heads,)."""

    relations: LongTensor
    """The unique relation indices, shape: (num_unique_relations,)."""

    tails: LongTensor
    """The unique tail entity indices, shape: (num_unique_tails,)."""

    positives: LongTensor
    """The positive triples, in batch-local indices, shape: (num_positive_triples, 3)."""

    weights: NotRequired[FloatTensor]
    """Sample weights, shape: (num_unique_heads, num_unique_relations, num_unique_tails)."""


class LCWABatch(TypedDict):
    """A batch for LCWA training."""

    pairs: LongTensor
    target: FloatTensor

    weights: NotRequired[FloatTensor]
    """Sample weights."""


class SLCWABatch(TypedDict):
    """A batch for sLCWA training."""

    #: the positive triples, shape: (batch_size, 3)
    positives: LongTensor

    #: sample weights for the positive triples
    pos_weights: NotRequired[FloatTensor]

    #: the negative triples, shape: (batch_size, num_negatives_per_positive, 3)
    negatives: LongTensor

    #: filtering masks for negative triples, shape: (batch_size, num_negatives_per_positive)
    masks: NotRequired[BoolTensor]

    #: sample weights for the negative triples
    neg_weights: NotRequired[FloatTensor]


class GroupedSLCWABatch(TypedDict):
    """An sLCWA batch keeping negatives grouped by the corrupted position."""

    #: the positive triples, shape: (batch_size, 3)
    positives: LongTensor

    #: sample weights for the positive triples
    pos_weights: NotRequired[FloatTensor]

    #: the replacement IDs, keyed by corrupted target, shape: (batch_size, k_target)
    corruptions: dict[Target, LongTensor]

    #: filtering masks for negative triples, keyed by corrupted target, shape: (batch_size, k_target)
    masks: NotRequired[dict[Target, BoolTensor]]

    #: sample weights for the negatives, keyed by corrupted target
    neg_weights: NotRequired[dict[Target, FloatTensor]]


class Instances(data.Dataset[BatchType], ABC, Generic[BatchType]):
    """Base class for training instances."""

    @abstractmethod
    def __len__(self):
        """Get the number of instances."""
        raise NotImplementedError


class BaseBatchedSLCWAInstances(
    Instances[SLCWABatch | GroupedSLCWABatch], data.IterableDataset[SLCWABatch | GroupedSLCWABatch]
):
    """Pre-batched training instances for the sLCWA training loop.

    .. note::

        this class is intended to be used with automatic batching disabled, i.e., both parameters `batch_size` and
        `batch_sampler` of torch.utils.data.DataLoader` are set to `None`.
    """

    #: the batch size
    batch_size: int

    @update_docstring_with_resolver_keys(
        ResolverKey("negative_sampler", "pykeen.sampling.negative_sampler_resolver"),
        ResolverKey("loss_weighter", "pykeen.triples.weights.loss_weighter_resolver"),
    )
    def __init__(
        self,
        mapped_triples: MappedTriples,
        batch_size: int = 1,
        drop_last: bool = True,
        num_entities: int | None = None,
        num_relations: int | None = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        loss_weighter: HintOrType[LossWeighter] = None,
        loss_weighter_kwargs: OptionalKwargs = None,
        grouped: bool = False,
    ) -> None:
        """Initialize the dataset.

        :param mapped_triples: shape: (num_triples, 3) the mapped triples
        :param batch_size: the batch size
        :param drop_last: whether to drop the last (incomplete) batch
        :param num_entities: >0 the number of entities, passed to the negative sampler
        :param num_relations: >0 the number of relations, passed to the negative sampler
        :param negative_sampler: the negative sampler, or a hint thereof
        :param negative_sampler_kwargs: additional keyword-based parameters used to instantiate the negative sampler
        :param loss_weighter: The method to determine sample weights.
        :param loss_weighter_kwargs: Parameters for the method to determine sample weights.
        :param grouped: whether to keep the negative samples grouped by corrupted target instead of materialising
            them as dense triples. This requires a negative sampler which supports grouped corruption, cf.
            :data:`~pykeen.sampling.NegativeSampler.supports_grouped_corruption`.

        :raises ValueError: if `grouped` is `True`, but the negative sampler does not support grouped corruption.
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
        self.loss_weighter = loss_weighter_resolver.make_safe(loss_weighter, loss_weighter_kwargs)
        if grouped and not self.negative_sampler.supports_grouped_corruption:
            raise ValueError(
                f"grouped=True requires a negative sampler which supports grouped corruption, but "
                f"{self.negative_sampler.__class__.__name__} does not. Consider using "
                f"{BasicNegativeSampler.__name__} instead."
            )
        self.grouped = grouped

    def _get_dense_batch(self, positive_batch: LongTensor) -> SLCWABatch:
        """Get a dense sLCWA batch for the given positive triples."""
        negative_batch, masks = self.negative_sampler.sample(positive_batch=positive_batch)
        result = SLCWABatch(positives=positive_batch, negatives=negative_batch)
        if masks is not None:
            result["masks"] = masks
        if self.loss_weighter is not None:
            result["pos_weights"] = self.loss_weighter.weight_triples(positive_batch)
            result["neg_weights"] = self.loss_weighter.weight_triples(negative_batch)
        return result

    def _get_grouped_batch(self, positive_batch: LongTensor) -> GroupedSLCWABatch:
        """Get a grouped sLCWA batch for the given positive triples."""
        corruptions, masks = self.negative_sampler.sample_grouped(positive_batch=positive_batch)
        result = GroupedSLCWABatch(positives=positive_batch, corruptions=corruptions)
        if masks is not None:
            result["masks"] = masks
        if self.loss_weighter is not None:
            h, r, t = positive_batch.unbind(dim=-1)
            result["pos_weights"] = self.loss_weighter(h=h, r=r, t=t)
            neg_weights: dict[Target, FloatTensor] = {}
            for target, replacements in corruptions.items():
                match target:
                    case pykeen_typing.LABEL_HEAD:
                        raw_weights = self.loss_weighter(h=replacements, r=r[:, None], t=t[:, None])
                    case pykeen_typing.LABEL_RELATION:
                        raw_weights = self.loss_weighter(h=h[:, None], r=replacements, t=t[:, None])
                    case pykeen_typing.LABEL_TAIL:
                        raw_weights = self.loss_weighter(h=h[:, None], r=r[:, None], t=replacements)
                # loss weighters may only depend on a subset of h/r/t (e.g. RelationLossWeighter ignores h/t) and
                # rely on __call__'s documented broadcasting semantics; broadcast explicitly to replacements' shape
                # so downstream concatenation/masking sees the same per-negative shape as the dense path. clone()
                # since broadcast_to returns a non-writable expanded view (e.g. incompatible with pin_memory).
                neg_weights[target] = raw_weights.broadcast_to(replacements.shape).clone()
            result["neg_weights"] = neg_weights
        return result

    def __getitem__(self, item: list[int]) -> SLCWABatch | GroupedSLCWABatch:
        """Get a batch from the given list of positive triple IDs."""
        positive_batch = self.mapped_triples[item]
        if self.grouped:
            return self._get_grouped_batch(positive_batch=positive_batch)
        return self._get_dense_batch(positive_batch=positive_batch)

    @abstractmethod
    def iter_triple_ids(self) -> Iterable[list[int]]:
        """Iterate over batches of IDs of positive triples."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[SLCWABatch | GroupedSLCWABatch]:
        """Iterate over batches."""
        for triple_ids in self.iter_triple_ids():
            yield self[triple_ids]

    def __len__(self) -> int:
        """Return the number of batches."""
        num_batches, remainder = divmod(len(self.mapped_triples), self.batch_size)
        if remainder and not self.drop_last:
            num_batches += 1
        return num_batches

    @classmethod
    def from_triples_factory(cls, tf: CoreTriplesFactory, create_inverse_triples: bool = False, **kwargs) -> Self:
        """Create sLCWA instances for triples factory.

        :param tf: The triples factory.
        :param create_inverse_triples:
            Whether to add inverse triples.
        :param kwargs: Additional keyword-based parameters passed to :meth:`__init__`

        :returns: The instances.
        """
        # TODO: can we better type `kwargs`?
        if "shuffle" in kwargs:
            if kwargs.pop("shuffle"):
                warnings.warn("Training instances are always shuffled.", DeprecationWarning, stacklevel=2)
            else:
                raise AssertionError("If shuffle is provided, it must be True.")
        if kwargs.pop("sampler", None):
            raise AssertionError("sampler is not handled in sLCWA instances")

        return cls(
            mapped_triples=tf._add_inverse_triples_if_necessary(
                mapped_triples=tf.mapped_triples, create_inverse_triples=create_inverse_triples
            ),
            num_entities=tf.num_entities,
            num_relations=2 * tf.real_num_relations if create_inverse_triples else tf.real_num_relations,
            **kwargs,
        )


class BatchedSLCWAInstances(BaseBatchedSLCWAInstances):
    """Random pre-batched training instances for the sLCWA training loop."""

    #: the number of iterations started in this (worker) process; used to vary the shared permutation across epochs
    #: when the data loader's workers are persistent (and thus their base seed does not change between epochs).
    #: note: this class-level default is shadowed by an instance attribute upon the first increment. This relies on
    #: each worker process operating on its own copy of the dataset, and starting its iterator exactly once per epoch,
    #: such that all workers share the same counter value in each epoch.
    _num_worker_iterations: int = 0

    def iter_triple_ids(self) -> Iterable[list[int]]:  # noqa: D102
        worker_info = data.get_worker_info()
        if worker_info is None:
            # single-process data loading
            yield from data.BatchSampler(
                sampler=data.RandomSampler(data_source=range(len(self.mapped_triples))),
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            return

        # multi-process data loading: to obtain the same batches (in particular, the same number of batches, and at
        # most one incomplete batch) as for single-process loading, all workers generate the same random permutation
        # of *all* triple IDs, split it into batches, and each worker only yields its share of these batches.
        # The workers' seeds are `base_seed + worker_id`, where `base_seed` is drawn anew for each data loader
        # iterator, cf. https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
        # This means for each epoch, unless `persistent_workers=True`, in which case the workers (and their seeds) are
        # reused across epochs; hence, we additionally mix in the number of iterations started by this worker.
        base_seed = worker_info.seed - worker_info.id
        generator = np.random.default_rng([base_seed, self._num_worker_iterations])
        self._num_worker_iterations += 1
        permutation = generator.permutation(len(self.mapped_triples))
        # batch b consists of the triple IDs permutation[b * batch_size : (b + 1) * batch_size]; __len__ already
        # accounts for dropping an incomplete last batch.
        # note: with `in_order=True` (the default), the data loader fetches from the workers in a round-robin fashion;
        # hence assigning batches round-robin, too, retains the batch order of the shared permutation. With
        # `in_order=False`, the batches' contents are still the same, but their order may differ.
        for batch_id in range(worker_info.id, len(self), worker_info.num_workers):
            yield permutation[batch_id * self.batch_size : (batch_id + 1) * self.batch_size].tolist()


class SubGraphSLCWAInstances(BaseBatchedSLCWAInstances):
    """Pre-batched training instances for SLCWA of coherent subgraphs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the instances.

        :param kwargs: keyword-based parameters passed to :meth:`BaseBatchedSLCWAInstances.__init__`
        """
        super().__init__(**kwargs)
        # indexing
        self.degrees, self.offset, self.neighbors = compute_compressed_adjacency_list(
            mapped_triples=self.mapped_triples
        )
        # numpy versions for the scalar-heavy sampling loop, cf. subgraph_sample. note: these need to be copies rather
        # than views, since pickling for data loader workers may move the tensors' storage into shared memory.
        self._degrees: np.ndarray = self.degrees.numpy().copy()
        self._offset: np.ndarray = self.offset.numpy().copy()
        self._entry_edge: np.ndarray = self.neighbors[:, 0].numpy().copy()
        self._entry_other: np.ndarray = self.neighbors[:, 1].numpy().copy()
        self._non_isolated: np.ndarray = np.flatnonzero(self._degrees > 0)

    def _sample_unvisited_vertex(self, generator: np.random.Generator, visited: set[int]) -> int:
        """Sample a vertex uniformly among the not yet visited vertices with at least one incident edge."""
        candidates = self._non_isolated
        # rejection sampling, which is efficient as long as only a small fraction of vertices has been visited
        for _ in range(32):
            vertex = int(candidates[generator.integers(len(candidates))])
            if vertex not in visited:
                return vertex
        # exact fallback
        candidates = np.setdiff1d(candidates, np.fromiter(visited, dtype=candidates.dtype, count=len(visited)))
        if not len(candidates):
            # cannot happen when requesting at most as many edges as there are, cf. subgraph_sample
            raise ValueError("There is no unvisited vertex with incident edges left.")
        return int(candidates[generator.integers(len(candidates))])

    def subgraph_sample(self, *, size: int | None = None, generator: np.random.Generator | None = None) -> list[int]:
        """Sample one subgraph.

        The subgraph is grown iteratively: in each step, a vertex is chosen among the visited
        vertices with probability proportional to its number of not yet picked incident edges, and
        one of these edges is chosen uniformly at random. If no visited vertex has any remaining
        incident edges, a not yet visited vertex is chosen uniformly at random (among those with at
        least one incident edge) instead. The other end of the chosen edge becomes visited, too.

        Choosing a vertex proportionally to its number of remaining incident edges, and then one of
        these uniformly, is the same as choosing uniformly among all remaining *half-edges* (i.e.,
        adjacency list entries) of visited vertices. Thus, we maintain a pool of half-edges of
        visited vertices, from which we draw uniformly. Half-edges whose edge has already been
        picked via its other half are removed lazily when drawn. Since every draw removes one entry
        from the pool, and each picked edge leaves at most one stale half-edge, sampling a batch
        requires at most `2 * size` half-edge draws.

        :param size: the number of edges to sample; defaults to :attr:`batch_size`. If it exceeds
            the number of triples, all triples are returned (in the order in which the sampling
            process picks them).
        :param generator: a NumPy generator. If not given, is derived from torch's global RNG, such
            that sampling is reproducible via :func:`torch.manual_seed`, and data loader worker
            processes, which PyTorch seeds differently, obtain different streams

        :returns: the triple IDs of the subgraph's edges, a list of `min(size, num_triples)` unique
            IDs

        :raises ValueError: if `size` is negative
        """
        if size is None:
            size = self.batch_size
        if size < 0:
            raise ValueError(f"size must be non-negative, but is {size}.")
        size = min(size, len(self.mapped_triples))

        if generator is None:
            generator = np.random.default_rng(int(torch.randint(2**62, size=()).item()))

        degrees, offset, entry_edge, entry_other = self._degrees, self._offset, self._entry_edge, self._entry_other

        # pool[:pool_size] contains the indices of the adjacency list entries of the visited vertices which have not
        # been drawn yet
        pool = np.empty_like(entry_edge)
        pool_size = 0
        visited: set[int] = set()
        picked: set[int] = set()
        result: list[int] = []
        # draw uniform random numbers in chunks, since per-call overhead dominates for single draws
        uniforms: list[float] = []
        while len(result) < size:
            if not pool_size:
                vertex = self._sample_unvisited_vertex(generator=generator, visited=visited)
                visited.add(vertex)
                start = offset[vertex]
                pool_size = int(degrees[vertex])
                pool[:pool_size] = np.arange(start, start + pool_size)

            # choose a half-edge uniformly
            if not uniforms:
                uniforms = generator.random(size=size).tolist()
            i = min(int(uniforms.pop() * pool_size), pool_size - 1)
            entry = pool[i]
            # remove it from the pool by swapping in the last element: it is either stale, or gets picked now
            pool_size -= 1
            pool[i] = pool[pool_size]
            edge = int(entry_edge[entry])
            if edge in picked:
                continue
            picked.add(edge)
            result.append(edge)

            # visit the other end
            other = int(entry_other[entry])
            if other not in visited:
                visited.add(other)
                start = offset[other]
                degree = int(degrees[other])
                pool[pool_size : pool_size + degree] = np.arange(start, start + degree)
                pool_size += degree
        return result

    def iter_triple_ids(self) -> Iterable[list[int]]:  # noqa: D102
        yield from (self.subgraph_sample() for _ in split_workload(len(self)))


class LCWAInstances(Instances[LCWABatch]):
    """Triples and mappings to their indices for LCWA."""

    @update_docstring_with_resolver_keys(ResolverKey("loss_weighter", "pykeen.triples.weights.loss_weighter_resolver"))
    def __init__(
        self,
        *,
        pairs: np.ndarray,
        compressed: scipy.sparse.csr_matrix,
        target: TargetHint = None,
        loss_weighter: HintOrType[LossWeighter] = None,
        loss_weighter_kwargs: OptionalKwargs = None,
    ) -> None:
        """Initialize the LCWA instances.

        :param pairs: The unique pairs
        :param compressed: The compressed triples in CSR format
        :param target: The prediction target.
        :param loss_weighter: The method to determine sample weights.
        :param loss_weighter_kwargs: Parameters for the method to determine sample weights.
        """
        self.pairs = pairs
        self.compressed = compressed
        self.loss_weighter = loss_weighter_resolver.make_safe(loss_weighter, loss_weighter_kwargs)
        self.target: TargetColumn = get_target_column(target=target)

    @classmethod
    def from_triples(
        cls,
        mapped_triples: MappedTriples,
        *,
        num_entities: int,
        num_relations: int,
        target: TargetHint = None,
        loss_weighter: HintOrType[LossWeighter] = None,
        loss_weighter_kwargs: OptionalKwargs = None,
    ) -> Self:
        """Create LCWA instances from triples.

        :param mapped_triples: shape: (num_triples, 3) The ID-based triples.
        :param num_entities: The number of entities.
        :param num_relations: The number of relations.
        :param target: The column to predict
        :param loss_weighter: The method to determine sample weights.
        :param loss_weighter_kwargs: Parameters for the method to determine sample weights.

        :returns: The instances.
        """
        target = get_target_column(target)
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
        return cls(
            pairs=unique_pairs,
            compressed=compressed,
            target=target,
            loss_weighter=loss_weighter,
            loss_weighter_kwargs=loss_weighter_kwargs,
        )

    @classmethod
    def from_triples_factory(
        cls,
        tf: CoreTriplesFactory,
        *,
        create_inverse_triples: bool = False,
        target: TargetHint = None,
        loss_weighter: HintOrType[LossWeighter] = None,
        loss_weighter_kwargs: OptionalKwargs = None,
    ) -> Self:
        """Create LCWA instances for triples factory.

        :param tf: The triples factory.
        :param create_inverse_triples:
            Whether to add inverse triples.
        :param target: The column to predict
        :param loss_weighter: The method to determine sample weights.
        :param loss_weighter_kwargs: Parameters for the method to determine sample weights.

        :returns: The instances.
        """
        return cls.from_triples(
            mapped_triples=tf._add_inverse_triples_if_necessary(
                mapped_triples=tf.mapped_triples,
                create_inverse_triples=create_inverse_triples,
            ),
            num_entities=tf.num_entities,
            num_relations=2 * tf.real_num_relations if create_inverse_triples else tf.real_num_relations,
            target=target,
            loss_weighter=loss_weighter,
            loss_weighter_kwargs=loss_weighter_kwargs,
        )

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, item: int) -> LCWABatch:
        pairs = self.pairs[item]
        result = LCWABatch(pairs=pairs, target=torch.from_numpy(np.asarray(self.compressed[item, :].todense())[0, :]))
        if self.loss_weighter is None:
            return result
        x = pairs[..., None, 0]
        y = pairs[..., None, 1]
        match self.target:
            # note: we need qualification here
            case pykeen_typing.COLUMN_HEAD:
                result["weights"] = self.loss_weighter(h=None, r=x, t=y)
            case pykeen_typing.COLUMN_RELATION:
                result["weights"] = self.loss_weighter(h=x, r=None, t=y)
            case pykeen_typing.COLUMN_TAIL:
                result["weights"] = self.loss_weighter(h=x, r=y, t=None)
        return result
