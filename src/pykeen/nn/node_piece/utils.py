"""Utilities for NodePiece."""

import logging
from typing import Collection, Mapping, Optional

import numpy
import scipy.sparse
import torch
from tqdm.auto import tqdm

__all__ = [
    "page_rank",
    "edge_index_to_sparse_matrix",
    "random_sample_no_replacement",
]

logger = logging.getLogger(__name__)


def page_rank(
    edge_index: numpy.ndarray,
    max_iter: int = 1_000,
    alpha: float = 0.05,
    epsilon: float = 1.0e-04,
    x0: Optional[numpy.ndarray] = None,
) -> numpy.ndarray:
    """
    Compute page-rank vector by power iteration.

    :param edge_index: shape: (2, m)
        the edge index of the graph, i.e, the edge list.
    :param max_iter: $>0$
        the maximum number of iterations
    :param alpha: $0 < x < 1$
        the smoothing value / teleport probability
    :param epsilon: $>0$
        a (small) constant to check for convergence
    :param x0:
        the initial value for $x$. If None, set to a constant $1/n$ vector.

    :return: shape: (n,)
        the page-rank vector, i.e., a score between 0 and 1 for each node.
    """
    # convert to sparse matrix, shape: (n, n)
    adj = edge_index_to_sparse_matrix(edge_index=edge_index)
    # symmetrize
    adj = adj + adj.transpose()
    # TODO: should we add self-links
    # adj = adj + scipy.sparse.eye(m=adj.shape[0], format="coo")
    # convert to CSR
    adj = adj.tocsr()
    # adjacency normalization
    degree_inv = numpy.reciprocal(numpy.asarray(adj.sum(axis=0), dtype=float))[0]
    adj = adj.dot(scipy.sparse.diags(degree_inv))

    # input normalization
    if x0 is None:
        n = adj.shape[0]
        x0 = numpy.full(shape=(n,), fill_value=1.0 / n)

    # power iteration
    x_old = x = x0
    beta = 1.0 - alpha
    for i in range(max_iter):
        x = beta * adj.dot(x) + alpha * x0
        if numpy.linalg.norm(x - x_old, ord=float("+inf")) < epsilon:
            logger.debug(f"Converged after {i} iterations up to {epsilon}.")
            break
        x_old = x
    else:  # for/else, cf. https://book.pythontips.com/en/latest/for_-_else.html
        logger.warning(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")
    return x


def edge_index_to_sparse_matrix(
    edge_index: numpy.ndarray,
    num_entities: Optional[int] = None,
) -> scipy.sparse.spmatrix:
    """Convert an edge index to a sparse matrix."""
    if num_entities is None:
        num_entities = edge_index.max().item() + 1
    return scipy.sparse.coo_matrix(
        (
            numpy.ones_like(edge_index[0], dtype=bool),
            tuple(edge_index),
        ),
        shape=(num_entities, num_entities),
    )


def random_sample_no_replacement(
    pool: Mapping[int, Collection[int]],
    num_tokens: int,
) -> torch.LongTensor:
    """Sample randomly without replacement num_tokens relations for each entity."""
    assignment = torch.full(
        size=(len(pool), num_tokens),
        dtype=torch.long,
        fill_value=-1,
    )
    # TODO: vectorization?
    for idx, this_pool in tqdm(pool.items(), desc="sampling", leave=False, unit_scale=True):
        this_pool_t = torch.as_tensor(data=list(this_pool), dtype=torch.long)
        this_pool = this_pool_t[torch.randperm(this_pool_t.shape[0])[:num_tokens]]
        assignment[idx, : len(this_pool_t)] = this_pool
    return assignment
