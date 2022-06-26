# -*- coding: utf-8 -*-

"""Anchor search for NodePiece."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy
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
    # Resolver
    "anchor_searcher_resolver",
    # Base classes
    "AnchorSearcher",
    # Concrete classes
    "ScipySparseAnchorSearcher",
    "CSGraphAnchorSearcher",
    "SparseBFSSearcher",
    "PersonalizedPageRankAnchorSearcher",
]

logger = logging.getLogger(__name__)


class AnchorSearcher(ExtraReprMixin, ABC):
    """A method for finding the closest anchors."""

    @abstractmethod
    def __call__(
        self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int, num_entities: Optional[int] = None
    ) -> numpy.ndarray:
        """
        Find the $k$ closest anchor nodes for each entity.

        :param edge_index: shape: (2, m)
            the edge index
        :param anchors: shape: (a,)
            the selected anchor entity Ids
        :param k:
            the number of closest anchors to return
        :param num_entities:
            the number of entities

        :return: shape: (n, k), -1 <= res < a
            the Ids of the closest anchors
        """
        raise NotImplementedError


class CSGraphAnchorSearcher(AnchorSearcher):
    """Find closest anchors using :class:`scipy.sparse.csgraph`."""

    # docstr-coverage: inherited
    def __call__(
        self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int, num_entities: Optional[int] = None
    ) -> numpy.ndarray:  # noqa: D102
        # convert to adjacency matrix
        adjacency = edge_index_to_sparse_matrix(edge_index=torch.as_tensor(edge_index, dtype=torch.long)).coalesce()
        # convert to scipy sparse csr
        adjacency = scipy.sparse.coo_matrix((adjacency.values(), adjacency.indices()), shape=adjacency.shape).tocsr()
        # compute distances between anchors and all nodes, shape: (num_anchors, num_entities)
        distances = scipy.sparse.csgraph.shortest_path(
            csgraph=adjacency,
            directed=False,
            return_predecessors=False,
            unweighted=True,
            indices=anchors,
        )
        # TODO: padding for unreachable?
        # select anchor IDs with smallest distance
        return torch.as_tensor(
            numpy.argpartition(distances, kth=min(k, distances.shape[0] - 1), axis=0)[:k, :].T,
            dtype=torch.long,
        )


class ScipySparseAnchorSearcher(AnchorSearcher):
    """Find closest anchors using :mod:`scipy.sparse`."""

    def __init__(self, max_iter: int = 5) -> None:
        """
        Initialize the searcher.

        :param max_iter:
            the maximum number of hops to consider
        """
        self.max_iter = max_iter

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"max_iter={self.max_iter}"

    @staticmethod
    def create_adjacency(edge_index: numpy.ndarray, num_entities: Optional[int] = None) -> scipy.sparse.spmatrix:
        """
        Create a sparse adjacency matrix from a given edge index.

        :param edge_index: shape: (2, m)
            the edge index
        :param num_entities:
            the number of entities. Can be inferred from `edge_index`

        :return: shape: (n, n)
            a square sparse adjacency matrix
        """
        # infer shape
        num_entities = ensure_num_entities(edge_index, num_entities=num_entities)
        # create adjacency matrix
        adjacency = scipy.sparse.coo_matrix(
            (
                numpy.ones_like(edge_index[0], dtype=bool),
                tuple(edge_index),
            ),
            shape=(num_entities, num_entities),
        )
        # symmetric + self-loops
        adjacency = adjacency + adjacency.transpose() + scipy.sparse.eye(num_entities, dtype=bool, format="coo")
        adjacency = adjacency.tocsr()
        logger.debug(
            f"Created sparse adjacency matrix of shape {adjacency.shape} where "
            f"{format_relative_comparison(part=adjacency.nnz, total=numpy.prod(adjacency.shape))} "
            f"are non-zero entries.",
        )
        return adjacency

    @staticmethod
    def bfs(
        anchors: numpy.ndarray,
        adjacency: scipy.sparse.spmatrix,
        max_iter: int,
        k: int,
    ) -> numpy.ndarray:
        """
        Determine the candidate pool using breadth-first search.

        :param anchors: shape: (a,)
            the anchor node IDs
        :param adjacency: shape: (n, n)
            the adjacency matrix
        :param max_iter:
            the maximum number of hops to consider
        :param k:
            the minimum number of anchor nodes to reach

        :return: shape: (n, a)
            a boolean array indicating whether anchor $j$ is in the set of $k$ closest anchors for node $i$
        """
        num_entities = adjacency.shape[0]
        # for each entity, determine anchor pool by BFS
        num_anchors = len(anchors)

        # an array storing whether node i is reachable by anchor j
        reachable = numpy.zeros(shape=(num_entities, num_anchors), dtype=bool)
        reachable[anchors] = numpy.eye(num_anchors, dtype=bool)

        # an array indicating whether a node is closed, i.e., has found at least $k$ anchors
        final = numpy.zeros(shape=(num_entities,), dtype=bool)

        # the output
        pool = numpy.zeros(shape=(num_entities, num_anchors), dtype=bool)
        # anchor nodes have themselves as a starting found anchor
        pool[anchors] = numpy.eye(num_anchors, dtype=bool)

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
        pool: numpy.ndarray,
        k: int,
    ) -> numpy.ndarray:
        """
        Select $k$ anchors from the given pools.

        :param pool: shape: (n, a)
            the anchor candidates for each node (a binary array)
        :param k:
            the number of candidates to select

        :return: shape: (n, k)
            the selected anchors. May contain -1 if there is an insufficient number of  candidates
        """
        tokens = numpy.full(shape=(pool.shape[0], k), fill_value=-1, dtype=int)
        generator = numpy.random.default_rng()
        # TODO: can we replace this loop with something vectorized?
        for i, row in enumerate(pool):
            (this_pool,) = row.nonzero()
            chosen = generator.choice(a=this_pool, size=min(k, this_pool.size), replace=False, shuffle=False)
            tokens[i, : len(chosen)] = chosen
        return tokens

    # docstr-coverage: inherited
    def __call__(
        self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int, num_entities: Optional[int] = None
    ) -> numpy.ndarray:  # noqa: D102
        adjacency = self.create_adjacency(edge_index=edge_index, num_entities=num_entities)
        pool = self.bfs(anchors=anchors, adjacency=adjacency, max_iter=self.max_iter, k=k)
        return self.select(pool=pool, k=k)


class SparseBFSSearcher(AnchorSearcher):
    """Find closest anchors using :mod:`torch_sparse` on a GPU."""

    def __init__(self, max_iter: int = 5, device: DeviceHint = None):
        """Initialize the tokenizer.

        :param max_iter:
            the number of partitions obtained through Metis.
        :param device:
            the device to use for tokenization
        """
        self.max_iter = max_iter
        self.device = resolve_device(device)

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"max_iter={self.max_iter}"

    @staticmethod
    def create_adjacency(
        edge_index: numpy.ndarray,
        num_entities: Optional[int] = None,
    ) -> torch.tensor:
        """
        Create a sparse adjacency matrix (in the form of the edge list) from a given edge index.

        :param edge_index: shape: (2, m)
            the edge index
        :param num_entities:
            The number of entities. If not given, inferred from the edge index

        :return: shape: (2, 2m + n)
            edge list with inverse edges and self-loops
        """
        num_entities = ensure_num_entities(edge_index, num_entities=num_entities)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        # symmetric + self-loops
        edge_list = torch.cat(
            [edge_index, edge_index.flip(0), torch.arange(num_entities).unsqueeze(0).repeat(2, 1)], dim=-1
        ).unique(
            dim=1
        )  # unique for deduplicating repeated edges

        return edge_list

    @staticmethod
    def bfs(
        anchors: numpy.ndarray,
        edge_list: torch.tensor,
        max_iter: int,
        k: int,
        device: torch.device,
    ) -> numpy.ndarray:
        """
        Determine the candidate pool using breadth-first search.

        :param anchors: shape: (a,)
            the anchor node IDs
        :param edge_list: shape: (2, n)
            the edge list with symmetric edges and self-loops
        :param max_iter:
            the maximum number of hops to consider
        :param k:
            the minimum number of anchor nodes to reach
        :param device:
            the device on which the calculations are done

        :return: shape: (n, a)
            a boolean array indicating whether anchor $j$ is in the set of $k$ closest anchors for node $i$

        :raises ImportError:
            If :mod:`torch_sparse` is not installed
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
        pool: torch.tensor,
        k: int,
    ) -> numpy.ndarray:
        """
        Select $k$ anchors from the given pools.

        :param pool: shape: (n, a)
            the anchor candidates for each node with distances
        :param k:
            the number of candidates to select

        :return: shape: (n, k)
            the selected anchors. May contain -1 if there is an insufficient number of  candidates
        """
        # sort the pool by nearest to farthest anchors
        values, indices = torch.sort(pool, dim=-1)
        # values with distance 255 (or max for unsigned int8 type) are padding tokens
        indices[values == torch.iinfo(values.dtype).max] = -1
        # since the output is sorted, no need for random sampling, we just take top-k nearest
        tokens = indices[:, :k].detach().cpu().numpy()
        return tokens

    # docstr-coverage: inherited
    def __call__(
        self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int, num_entities: Optional[int] = None
    ) -> numpy.ndarray:  # noqa: D102
        edge_list = self.create_adjacency(edge_index=edge_index, num_entities=num_entities)
        pool = self.bfs(anchors=anchors, edge_list=edge_list, max_iter=self.max_iter, k=k, device=self.device)
        return self.select(pool=pool, k=k)


class PersonalizedPageRankAnchorSearcher(AnchorSearcher):
    """
    Select closest anchors as the nodes with the largest personalized page rank.

    .. seealso::
        http://web.stanford.edu/class/cs224w/slides/04-pagerank.pdf
    """

    def __init__(self, batch_size: int = 1, use_tqdm: bool = False, page_rank_kwargs: OptionalKwargs = None):
        """
        Initialize the searcher.

        :param batch_size:
            the batch size to use.
        :param use_tqdm:
            whether to use tqdm
        :param page_rank_kwargs:
            keyword-based parameters used for :func:`page_rank`. Must not include `edge_index`, or `x0`.
        """
        self.batch_size = batch_size
        self.page_rank_kwargs = page_rank_kwargs or {}
        self.use_tqdm = use_tqdm

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield f"batch_size={self.batch_size}"
        yield f"use_tqdm={self.use_tqdm}"
        yield f"page_rank_kwargs={self.page_rank_kwargs}"

    def precalculate_anchor_ppr(self, edge_index: numpy.ndarray, anchors: numpy.ndarray) -> numpy.ndarray:
        """
        Sort anchors nodes by PPR values from each node.

        :param edge_index: shape: (2, m)
            the edge index.
        :param anchors: shape: `(num_anchors,)`
            the anchor IDs.

        :return: shape: `(num_entities, num_anchors)`
            the PPR values for each anchor
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

    # docstr-coverage: inherited
    def __call__(
        self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int, num_entities: Optional[int] = None
    ) -> numpy.ndarray:  # noqa: D102
        num_entities = ensure_num_entities(edge_index, num_entities=num_entities)
        result = numpy.full(shape=(num_entities, k), fill_value=-1)
        i = 0
        for batch_ppr in self._iter_ppr(edge_index=edge_index, anchors=anchors, num_entities=num_entities):
            batch_size = batch_ppr.shape[0]
            # select k anchors with largest ppr, shape: (batch_size, k)
            result[i : i + batch_size, :] = torch.topk(batch_ppr, k=k, dim=-1, largest=True).indices.cpu().numpy()
            i += batch_size
        return result

    @torch.inference_mode()
    def _iter_ppr(
        self, edge_index: numpy.ndarray, anchors: numpy.ndarray, num_entities: Optional[int] = None
    ) -> Iterable[torch.Tensor]:
        """
        Yield batches of PPR values for each anchor from each entities' perspective.

        :param edge_index: shape: (2, m)
            the edge index.
        :param anchors: shape: `(num_anchors,)`
            the anchor IDs.
        :param num_entities:
            The number of entities. Will be calculated on-the-fly if not given

        :yields: shape: (batch_size, num_anchors)
            batches of anchor PPRs.
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
        anchors = torch.as_tensor(anchors, dtype=torch.long)
        for start in progress:
            # run page-rank calculation, shape: (batch_size, n)
            ppr = page_rank(
                adj=adj, x0=prepare_x0(indices=range(start, start + self.batch_size), n=n), **self.page_rank_kwargs
            )
            # select PPR values for the anchors, shape: (batch_size, num_anchors)
            yield ppr[:, anchors.to(ppr.device)]


anchor_searcher_resolver: ClassResolver[AnchorSearcher] = ClassResolver.from_subclasses(
    base=AnchorSearcher,
    default=CSGraphAnchorSearcher,
)
