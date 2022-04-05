# -*- coding: utf-8 -*-

"""Anchor search for NodePiece."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable

import numpy
import scipy.sparse
import torch
from class_resolver import ClassResolver

from .utils import edge_index_to_sparse_matrix
from ...utils import format_relative_comparison

__all__ = [
    # Resolver
    "anchor_searcher_resolver",
    # Base classes
    "AnchorSearcher",
    # Concrete classes
    "ScipySparseAnchorSearcher",
    "CSGraphAnchorSearcher",
]

logger = logging.getLogger(__name__)


class AnchorSearcher(ABC):
    """A method for finding the closest anchors."""

    @abstractmethod
    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:
        """
        Find the $k$ closest anchor nodes for each entity.

        :param edge_index: shape: (2, m)
            the edge index
        :param anchors: shape: (a,)
            the selected anchor entity Ids
        :param k:
            the number of closest anchors to return

        :return: shape: (n, k), -1 <= res < a
            the Ids of the closest anchors
        """
        raise NotImplementedError

    def extra_repr(self) -> Iterable[str]:
        """Extra components for __repr__."""
        return []

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({', '.join(self.extra_repr())})"


class CSGraphAnchorSearcher(AnchorSearcher):
    """Find closest anchors using :class:`scipy.sparse.csgraph`."""

    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:  # noqa: D102
        # convert to adjacency matrix
        adjacency = edge_index_to_sparse_matrix(edge_index=edge_index).tocsr()
        # compute distances between anchors and all nodes, shape: (num_anchors, num_entities)
        distances = scipy.sparse.csgraph.shortest_path(
            csgraph=adjacency,
            directed=False,
            return_predecessors=False,
            unweighted=True,
            indices=anchors,
        )
        # select anchor IDs with smallest distance
        return torch.as_tensor(
            numpy.argpartition(distances, kth=min(k, distances.shape[0]), axis=0)[:k, :].T,
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

    def extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().extra_repr()
        yield f"max_iter={self.max_iter}"

    @staticmethod
    def create_adjacency(
        edge_index: numpy.ndarray,
    ) -> scipy.sparse.spmatrix:
        """
        Create a sparse adjacency matrix from a given edge index.

        :param edge_index: shape: (2, m)
            the edge index

        :return: shape: (n, n)
            a square sparse adjacency matrix
        """
        # infer shape
        num_entities = edge_index.max().item() + 1
        # create adjacency matrix
        adjacency = scipy.sparse.coo_matrix(
            (
                numpy.ones_like(edge_index[0], dtype=bool),
                tuple(edge_index),
            ),
            shape=(num_entities, num_entities),
            dtype=bool,
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

    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:  # noqa: D102
        adjacency = self.create_adjacency(edge_index=edge_index)
        pool = self.bfs(anchors=anchors, adjacency=adjacency, max_iter=self.max_iter, k=k)
        return self.select(pool=pool, k=k)


anchor_searcher_resolver: ClassResolver[AnchorSearcher] = ClassResolver.from_subclasses(
    base=AnchorSearcher,
    default=CSGraphAnchorSearcher,
)
