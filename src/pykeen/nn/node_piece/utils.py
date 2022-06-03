"""Utilities for NodePiece."""

import logging
from typing import Collection, Mapping, Optional

import numpy
import scipy.sparse
import torch
from tqdm.auto import tqdm

__all__ = [
    "page_rank",
    "prepare_page_rank_adjacency",
    "edge_index_to_sparse_matrix",
    "random_sample_no_replacement",
]

logger = logging.getLogger(__name__)


def page_rank(
    adj: Optional[scipy.sparse.csr_matrix] = None,
    edge_index: Optional[numpy.ndarray] = None,
    max_iter: int = 1_000,
    alpha: float = 0.05,
    epsilon: float = 1.0e-04,
    x0: Optional[numpy.ndarray] = None,
    use_tqdm: bool = False,
) -> numpy.ndarray:
    """
    Compute page-rank vector by power iteration.

    :param adj:
        the adjacency matrix, cf. :func:`prepare_page_rank_adjacency`. Preferred over `edge_index`.
    :param edge_index: shape: (2, m)
        the edge index of the graph, i.e, the edge list.
    :param max_iter: $>0$
        the maximum number of iterations
    :param alpha: $0 < x < 1$
        the smoothing value / teleport probability
    :param epsilon: $>0$
        a (small) constant to check for convergence
    :param x0: shape: `(n,)`, or `(n, batch_size)`
        the initial value for $x$. If None, set to a constant $1/n$ vector.
    :param use_tqdm:
        whether to use a tqdm progress bar

    :return: shape: `(n,)`
        the page-rank vector, i.e., a score between 0 and 1 for each node.

    :raises ValueError:
        if neither `adj` nor `edge_index` are provided
    """
    if adj is None:
        if edge_index is None:
            raise ValueError("Must provide at least one of `adj` and `edge_index`.")
        adj = prepare_page_rank_adjacency(edge_index)

    # input normalization
    if x0 is None:
        n = adj.shape[0]
        x0 = numpy.full(shape=(n,), fill_value=1.0 / n)
    else:
        numpy.testing.assert_allclose(x0.sum(axis=0), 1.0)
        assert (x0 >= 0.0).all()
        assert (x0 <= 1.0).all()

    # power iteration
    x_old = x = x0
    beta = 1.0 - alpha
    progress = range(max_iter)
    if use_tqdm:
        progress = tqdm(progress, unit_scale=True, leave=False)
    for i in progress:
        x = beta * adj.dot(x) + alpha * x0
        max_diff = numpy.linalg.norm(x - x_old, ord=float("+inf"), axis=0).max()
        if use_tqdm:
            assert isinstance(progress, tqdm)  # for mypy
            progress.set_postfix(max_diff=max_diff)
        if max_diff < epsilon:
            logger.debug(f"Converged after {i} iterations up to {epsilon}.")
            break
        x_old = x
    else:  # for/else, cf. https://book.pythontips.com/en/latest/for_-_else.html
        logger.warning(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")
    return x


def prepare_page_rank_adjacency(edge_index: numpy.ndarray) -> scipy.sparse.csr_matrix:
    """
    Prepare the page-rank adjacency matrix.

    :param edge_index: shape: (2, n)
        the edge index

    :return:
        the symmetric, normalized, and sparse adjacency matrix
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
    return adj


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
