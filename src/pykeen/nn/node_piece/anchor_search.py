"""Anchor search for NodePiece."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import scipy.sparse
import torch
from class_resolver import ClassResolver, OptionalKwargs
from torch_ppr import page_rank
from torch_ppr.utils import edge_index_to_sparse_matrix, prepare_page_rank_adjacency, prepare_x0
from tqdm.auto import tqdm

from .utils import ensure_num_entities
from ...typing import DeviceHint
from ...utils import ExtraReprMixin, format_relative_comparison, resolve_device

__all__ = [
    "AnchorSearcher",
    "CSGraphAnchorSearcher",
    "PersonalizedPageRankAnchorSearcher",
    "ScipySparseAnchorSearcher",
    "SparseBFSSearcher",
    "anchor_searcher_resolver",
]

logger = logging.getLogger(__name__)


class AnchorSearcher(ExtraReprMixin, ABC):
    """A method for finding the closest anchors."""

    @abstractmethod
    def __call__(
        self, edge_index: np.ndarray, anchors: np.ndarray, k: int, num_entities: int | None = None
    ) -> np.ndarray:
        """Find the $k$ closest anchor nodes for each entity.

        :param edge_index: shape: (2, m) the edge index
        :param anchors: shape: (a,) the selected anchor entity Ids
        :param k: the number of closest anchors to return
        :param num_entities: the number of entities

        :returns: shape: (n, k), -1 <= res < a the Ids of the closest anchors
        """
        raise NotImplementedError


class CSGraphAnchorSearcher(AnchorSearcher):
    """Find closest anchors using :class:`scipy.sparse.csgraph`."""

    @staticmethod
    def topk_argsort(array: np.ndarray, k: int) -> np.ndarray:
        """Return the sorted top-k indices using argsort.

        Its complexity is $O(m * n log n)$.

        :param array: shape: (n, m) the array
        :param k: the value of $k$

        :returns: shape: (m, k) the indices of the $k$ smallest values sorted in descending order
        """
        return np.argsort(array, axis=0)[:k, :]

    @staticmethod
    def topk_argpartition(array: np.ndarray, k: int) -> np.ndarray:
        """Return the sorted top-k indices using argpartition.

        Its complexity is $O(m * (n + k log k))$.

        :param array: shape: (n, m) the array
        :param k: the value of $k$

        :returns: shape: (m, k) the indices of the $k$ smallest values sorted in descending order
        """
        # this array contains the indices of the k closest anchors nodes, but without guarantee that they are sorted
        top_k_indices = np.argpartition(array, kth=min(k, array.shape[0] - 1), axis=0)[:k, :]
        # now we want to sort these top-k entries, (O(k log k)) (and only those)
        top_dist = np.take_along_axis(arr=array, indices=top_k_indices, axis=0)
        return np.take_along_axis(arr=top_k_indices, indices=np.argsort(top_dist, axis=0), axis=0)

    def __init__(self, chunk_size: int | None = None, max_memory: int = 2**28) -> None:
        """Initialize the searcher.

        The shortest path distances are computed for chunks of anchors at a time, and merged into a running top-$k$.
        Thus, only a `(chunk_size, num_entities)` distance matrix has to be kept in memory instead of the full
        `(num_anchors, num_entities)` one.

        :param chunk_size: the number of anchors for which to compute shortest path distances at once. If `None`, it
            is derived from `max_memory`.
        :param max_memory: the (approximate) memory budget in bytes for a single chunk's float64 distance matrix. Only
            used if `chunk_size` is `None`.

        :raises ValueError: if `chunk_size` or `max_memory` are not positive
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, but is {chunk_size}")
        if max_memory <= 0:
            raise ValueError(f"max_memory must be positive, but is {max_memory}")
        self.chunk_size = chunk_size
        self.max_memory = max_memory

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"chunk_size={self.chunk_size}"
        yield f"max_memory={self.max_memory}"

    @staticmethod
    def _merge_top_k(best: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
        """Merge new candidates into the running top-$k$ (unsorted).

        :param best: shape: (k', n) the current $k' <= k$ smallest keys for each entity
        :param candidates: shape: (c, n) the new candidate keys; will be modified in-place
        :param k: the value of $k$

        :returns: shape: (min(k, k' + c), n) the $k$ smallest keys among both inputs, in arbitrary order
        """
        # reduce the chunk to its top-k first (in-place) to avoid copying the full chunk
        if candidates.shape[0] > k:
            candidates.partition(k - 1, axis=0)
            candidates = candidates[:k]
        merged = np.concatenate([best, candidates], axis=0)
        if merged.shape[0] > k:
            merged.partition(k - 1, axis=0)
            merged = merged[:k]
        return merged

    def __call__(  # noqa: D102
        self, edge_index: np.ndarray, anchors: np.ndarray, k: int, num_entities: int | None = None
    ) -> np.ndarray:
        # infer shape; entities without any edge only reach themselves (if they are an anchor)
        n = ensure_num_entities(edge_index, num_entities=num_entities)
        # convert to adjacency matrix
        adjacency = edge_index_to_sparse_matrix(
            edge_index=torch.as_tensor(edge_index, dtype=torch.long), num_nodes=n
        ).coalesce()
        # convert to scipy sparse csr
        adjacency = scipy.sparse.coo_matrix((adjacency.values(), adjacency.indices()), shape=adjacency.shape).tocsr()
        num_anchors = len(anchors)
        # we encode (distance, anchor ID) pairs as single float64 keys `distance * num_anchors + anchor_id`. This makes
        # all finite keys unique, i.e., ties in distance are deterministically broken by the smaller anchor ID,
        # independent of the chunk size. Since unweighted distances are integers < n, the keys are exactly
        # representable as long as n * num_anchors <= 2**53. Unreachable anchors have an infinite key.
        if n * num_anchors > 2**53:
            raise ValueError(f"Too many entities and anchors for exact key encoding: {n=:_}, {num_anchors=:_}")
        chunk_size = self.chunk_size or max(1, self.max_memory // (8 * max(n, 1)))
        # the running top-k keys, shape: (k', n)
        best = np.empty(shape=(0, n), dtype=np.float64)
        for start in range(0, num_anchors, chunk_size):
            stop = min(start + chunk_size, num_anchors)
            # compute distances between anchors and all nodes, shape: (chunk_size, num_entities)
            keys = scipy.sparse.csgraph.shortest_path(
                csgraph=adjacency,
                directed=False,
                return_predecessors=False,
                unweighted=True,
                indices=anchors[start:stop],
            )
            # encode, in-place
            keys *= num_anchors
            keys += np.arange(start, stop, dtype=np.float64)[:, None]
            best = self._merge_top_k(best=best, candidates=keys, k=k)
        # sort by distance (and anchor ID), shape: (n, min(k, num_anchors))
        best = np.sort(best.T, axis=1)
        # decode anchor IDs; unreachable anchors, and those exceeding the number of anchors, are padded with -1
        reachable = np.isfinite(best)
        result = np.full(shape=(n, k), fill_value=-1, dtype=np.int64)
        result[:, : best.shape[1]][reachable] = best[reachable].astype(np.int64) % num_anchors
        return result


class ScipySparseAnchorSearcher(AnchorSearcher):
    """Find closest anchors using :mod:`scipy.sparse`."""

    def __init__(self, max_iter: int = 5) -> None:
        """Initialize the searcher.

        :param max_iter: the maximum number of hops to consider
        """
        self.max_iter = max_iter

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"max_iter={self.max_iter}"

    @staticmethod
    def create_adjacency(edge_index: np.ndarray, num_entities: int | None = None) -> scipy.sparse.spmatrix:
        """Create a sparse adjacency matrix from a given edge index.

        :param edge_index: shape: (2, m) the edge index
        :param num_entities: the number of entities. Can be inferred from `edge_index`

        :returns: shape: (n, n) a square sparse adjacency matrix
        """
        # infer shape
        num_entities = ensure_num_entities(edge_index, num_entities=num_entities)
        # create adjacency matrix
        adjacency = scipy.sparse.coo_matrix(
            (
                np.ones_like(edge_index[0], dtype=bool),
                tuple(edge_index),
            ),
            shape=(num_entities, num_entities),
        )
        # symmetric + self-loops
        adjacency = adjacency + adjacency.transpose() + scipy.sparse.eye(num_entities, dtype=bool, format="coo")
        adjacency = adjacency.tocsr()
        logger.debug(
            f"Created sparse adjacency matrix of shape {adjacency.shape} where "
            f"{format_relative_comparison(part=adjacency.nnz, total=np.prod(adjacency.shape))} "
            f"are non-zero entries.",
        )
        return adjacency

    @staticmethod
    def bfs(
        anchors: np.ndarray,
        adjacency: scipy.sparse.spmatrix,
        max_iter: int,
        k: int,
    ) -> np.ndarray:
        """Determine the candidate pool using breadth-first search.

        :param anchors: shape: (a,) the anchor node IDs
        :param adjacency: shape: (n, n) the adjacency matrix
        :param max_iter: the maximum number of hops to consider
        :param k: the minimum number of anchor nodes to reach

        :returns: shape: (n, a) a boolean array indicating whether anchor $j$ is in the set of $k$ closest anchors for
            node $i$
        """
        num_entities = adjacency.shape[0]
        # for each entity, determine anchor pool by BFS
        num_anchors = len(anchors)

        # an array storing whether node i is reachable by anchor j
        reachable = np.zeros(shape=(num_entities, num_anchors), dtype=bool)
        reachable[anchors] = np.eye(num_anchors, dtype=bool)

        # an array indicating whether a node is closed, i.e., has found at least $k$ anchors
        final = np.zeros(shape=(num_entities,), dtype=bool)

        # the output
        pool = np.zeros(shape=(num_entities, num_anchors), dtype=bool)
        # anchor nodes have themselves as a starting found anchor
        pool[anchors] = np.eye(num_anchors, dtype=bool)

        # TODO: take all (q-1) hop neighbors before selecting from q-hop
        old_reachable = reachable
        for i in range(max_iter):
            # propagate one hop
            reachable = adjacency.dot(reachable)
            # convergence check
            if (reachable == old_reachable).all():
                logger.warning(f"Search converged after iteration {i} without all nodes being reachable.")
                break
            old_reachable = reachable
            # copy pool if we have seen enough anchors and have not yet stopped
            num_reachable = reachable.sum(axis=1)
            enough = num_reachable >= k
            mask = enough & ~final
            logger.debug(
                f"Iteration {i}: {format_relative_comparison(enough.sum(), total=num_entities)} closed nodes.",
            )
            pool[mask] = reachable[mask]
            # stop once we have enough
            final |= enough
            if final.all():
                break
        return pool

    @staticmethod
    def select(
        pool: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Select $k$ anchors from the given pools.

        :param pool: shape: (n, a) the anchor candidates for each node (a binary array)
        :param k: the number of candidates to select

        :returns: shape: (n, k) the selected anchors. May contain -1 if there is an insufficient number of candidates
        """
        tokens = np.full(shape=(pool.shape[0], k), fill_value=-1, dtype=int)
        generator = np.random.default_rng()
        # TODO: can we replace this loop with something vectorized?
        for i, row in enumerate(pool):
            (this_pool,) = row.nonzero()
            chosen = generator.choice(a=this_pool, size=min(k, this_pool.size), replace=False, shuffle=False)
            tokens[i, : len(chosen)] = chosen
        return tokens

    def __call__(  # noqa: D102
        self, edge_index: np.ndarray, anchors: np.ndarray, k: int, num_entities: int | None = None
    ) -> np.ndarray:
        adjacency = self.create_adjacency(edge_index=edge_index, num_entities=num_entities)
        pool = self.bfs(anchors=anchors, adjacency=adjacency, max_iter=self.max_iter, k=k)
        return self.select(pool=pool, k=k)


class SparseBFSSearcher(AnchorSearcher):
    """Find closest anchors using :mod:`torch_sparse` on a GPU."""

    def __init__(self, max_iter: int = 5, device: DeviceHint = None):
        """Initialize the tokenizer.

        :param max_iter: the number of partitions obtained through Metis.
        :param device: the device to use for tokenization
        """
        self.max_iter = max_iter
        self.device = resolve_device(device)

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"max_iter={self.max_iter}"

    @staticmethod
    def create_adjacency(
        edge_index: np.ndarray,
        num_entities: int | None = None,
    ) -> torch.Tensor:
        """Create a sparse adjacency matrix (in the form of the edge list) from a given edge index.

        :param edge_index: shape: (2, m) the edge index
        :param num_entities: The number of entities. If not given, inferred from the edge index

        :returns: shape: (2, 2m + n) edge list with inverse edges and self-loops
        """
        num_entities = ensure_num_entities(edge_index, num_entities=num_entities)
        edge_index_torch = torch.as_tensor(edge_index, dtype=torch.long)

        # symmetric + self-loops
        return torch.cat(
            [edge_index_torch, edge_index_torch.flip(0), torch.arange(num_entities).unsqueeze(0).repeat(2, 1)], dim=-1
        ).unique(dim=1)  # unique for deduplicating repeated edges

    @staticmethod
    def bfs(
        anchors: np.ndarray,
        edge_list: torch.Tensor,
        max_iter: int,
        k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Determine the candidate pool using breadth-first search.

        :param anchors: shape: (a,) the anchor node IDs
        :param edge_list: shape: (2, n) the edge list with symmetric edges and self-loops
        :param max_iter: the maximum number of hops to consider
        :param k: the minimum number of anchor nodes to reach
        :param device: the device on which the calculations are done

        :returns: shape: (n, a) a boolean array indicating whether anchor $j$ is in the set of $k$ closest anchors for
            node $i$

        :raises ImportError: If :mod:`torch_sparse` is not installed
        """
        try:
            import torch_sparse
        except ImportError as err:
            raise ImportError("Requires `torch_sparse` to be installed.") from err

        num_entities = edge_list.max().item() + 1
        # for each entity, determine anchor pool by BFS
        num_anchors = len(anchors)

        anchors = torch.tensor(anchors, dtype=torch.long, device=device)

        # an array storing whether node i is reachable by anchor j
        reachable = torch.zeros((num_entities, num_anchors), dtype=torch.bool, device=device)
        reachable[anchors] = torch.eye(num_anchors, dtype=torch.bool, device=device)

        # an array indicating whether a node is closed, i.e., has found at least $k$ anchors
        final = torch.zeros((num_entities,), dtype=torch.bool, device=device)

        # the output that track the distance to each found anchor
        # dtype is unsigned int 8 bit, so we initialize the maximum distance to 255 (or max default)
        dtype = torch.uint8
        pool = torch.zeros((num_entities, num_anchors), dtype=dtype, device=device).fill_(torch.iinfo(dtype).max)
        # initial anchors are 0-hop away from themselves
        pool[anchors, torch.arange(len(anchors), dtype=torch.long, device=device)] = 0

        edge_list = edge_list.to(device)
        values = torch.ones_like(edge_list[0], dtype=torch.bool, device=device)

        old_reachable = reachable
        for i in range(max_iter):
            # propagate one hop
            # TODO the float() trick for GPU result stability until the torch_sparse issue is resolved
            # https://github.com/rusty1s/pytorch_sparse/issues/243
            reachable = (
                torch_sparse.spmm(
                    index=edge_list, value=values.float(), m=num_entities, n=num_entities, matrix=reachable.float()
                )
                > 0.0
            )
            # convergence check
            if (reachable == old_reachable).all():
                logger.warning(f"Search converged after iteration {i} without all nodes being reachable.")
                break
            # newly reached is a mask that points to newly discovered anchors at this particular step
            # implemented as element-wise XOR (will only give True in 0 XOR 1 or 1 XOR 0)
            # in our case we enrich the set of found anchors, so we can only have values turning 0 to 1, eg 0 XOR 1
            newly_reached = reachable ^ old_reachable
            old_reachable = reachable
            # copy pool if we have seen enough anchors and have not yet stopped
            num_reachable = reachable.sum(axis=1)
            enough = num_reachable >= k
            logger.debug(
                f"Iteration {i}: {format_relative_comparison(enough.sum(), total=num_entities)} closed nodes.",
            )
            # update the value in the pool by the current hop value (we start from 0, so +1 be default)
            pool[newly_reached] = i + 1
            # stop once we have enough
            final |= enough
            if final.all():
                break

        return pool

    @staticmethod
    def select(
        pool: torch.Tensor,
        k: int,
    ) -> np.ndarray:
        """Select $k$ anchors from the given pools.

        :param pool: shape: (n, a) the anchor candidates for each node with distances
        :param k: the number of candidates to select

        :returns: shape: (n, k) the selected anchors. May contain -1 if there is an insufficient number of candidates
        """
        # sort the pool by nearest to farthest anchors
        values, indices = torch.sort(pool, dim=-1)
        # values with distance 255 (or max for unsigned int8 type) are padding tokens
        indices[values == torch.iinfo(values.dtype).max] = -1
        # since the output is sorted, no need for random sampling, we just take top-k nearest
        top_k = indices[:, :k].detach().cpu().numpy()
        # pad to k, if there are fewer than k anchors
        result = np.full(shape=(top_k.shape[0], k), fill_value=-1, dtype=np.int64)
        result[:, : top_k.shape[1]] = top_k
        return result

    def __call__(  # noqa: D102
        self, edge_index: np.ndarray, anchors: np.ndarray, k: int, num_entities: int | None = None
    ) -> np.ndarray:
        edge_list = self.create_adjacency(edge_index=edge_index, num_entities=num_entities)
        pool = self.bfs(anchors=anchors, edge_list=edge_list, max_iter=self.max_iter, k=k, device=self.device)
        return self.select(pool=pool, k=k)


class PersonalizedPageRankAnchorSearcher(AnchorSearcher):
    """Select closest anchors as the nodes with the largest personalized page rank.

    .. seealso::

        http://web.stanford.edu/class/cs224w/slides/04-pagerank.pdf
    """

    def __init__(self, batch_size: int = 1, use_tqdm: bool = False, page_rank_kwargs: OptionalKwargs = None):
        """Initialize the searcher.

        :param batch_size: the batch size to use.
        :param use_tqdm: whether to use tqdm
        :param page_rank_kwargs: keyword-based parameters used for :func:`torch_ppr.page_rank`. Must not include
            `edge_index`, or `x0`.
        """
        self.batch_size = batch_size
        self.page_rank_kwargs = page_rank_kwargs or {}
        self.use_tqdm = use_tqdm

    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield f"batch_size={self.batch_size}"
        yield f"use_tqdm={self.use_tqdm}"
        yield f"page_rank_kwargs={self.page_rank_kwargs}"

    def precalculate_anchor_ppr(self, edge_index: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Sort anchors nodes by PPR values from each node.

        :param edge_index: shape: (2, m) the edge index.
        :param anchors: shape: `(num_anchors,)` the anchor IDs.

        :returns: shape: `(num_entities, num_anchors)` the PPR values for each anchor
        """
        return (
            torch.cat(
                [
                    ppr_batch.argsort(dim=-1)
                    for ppr_batch in self._iter_ppr(
                        edge_index=edge_index,
                        anchors=anchors,
                    )
                ]
            )
            .flip(-1)
            .cpu()
            .numpy()
        )

    def __call__(  # noqa: D102
        self, edge_index: np.ndarray, anchors: np.ndarray, k: int, num_entities: int | None = None
    ) -> np.ndarray:
        num_entities = ensure_num_entities(edge_index, num_entities=num_entities)
        result = np.full(shape=(num_entities, k), fill_value=-1)
        # if there are fewer than k anchors, the remaining columns are padded with -1
        num_selected = min(k, len(anchors))
        i = 0
        for batch_ppr in self._iter_ppr(edge_index=edge_index, anchors=anchors, num_entities=num_entities):
            batch_size = batch_ppr.shape[0]
            # select k anchors with largest ppr, shape: (batch_size, num_selected)
            result[i : i + batch_size, :num_selected] = (
                torch.topk(batch_ppr, k=num_selected, dim=-1, largest=True).indices.cpu().numpy()
            )
            i += batch_size
        return result

    @torch.inference_mode()
    def _iter_ppr(
        self, edge_index: np.ndarray, anchors: np.ndarray, num_entities: int | None = None
    ) -> Iterable[torch.Tensor]:
        """Yield batches of PPR values for each anchor from each entities' perspective.

        :param edge_index: shape: (2, m) the edge index.
        :param anchors: shape: `(num_anchors,)` the anchor IDs.
        :param num_entities: The number of entities. Will be calculated on-the-fly if not given

        :yields: shape: (batch_size, num_anchors) batches of anchor PPRs.
        """
        # prepare adjacency matrix only once
        adj = prepare_page_rank_adjacency(
            edge_index=torch.as_tensor(edge_index, dtype=torch.long), num_nodes=num_entities
        )
        # prepare result
        n = adj.shape[0]
        # progress bar?
        progress = range(0, n, self.batch_size)
        if self.use_tqdm:
            progress = tqdm(progress, unit="batch", unit_scale=True)
        # batch-wise computation of PPR
        anchors_torch = torch.as_tensor(anchors, dtype=torch.long)
        for start in progress:
            # run page-rank calculation, shape: (batch_size, n)
            ppr = page_rank(
                adj=adj, x0=prepare_x0(indices=range(start, start + self.batch_size), n=n), **self.page_rank_kwargs
            )
            # select PPR values for the anchors, shape: (batch_size, num_anchors)
            yield ppr[:, anchors_torch.to(ppr.device)]


#: A resolver for NodePiece anchor searchers
anchor_searcher_resolver: ClassResolver[AnchorSearcher] = ClassResolver.from_subclasses(
    base=AnchorSearcher,  # type: ignore[type-abstract]
    default=CSGraphAnchorSearcher,
)
