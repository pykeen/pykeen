# -*- coding: utf-8 -*-

"""Utilities for non-parametric baseline models."""

from typing import Optional, Tuple

import numpy
import scipy.sparse
import torch
from sklearn.preprocessing import normalize as sklearn_normalize

from ...triples import CoreTriplesFactory
from ...triples.leakage import jaccard_similarity_scipy
from ...typing import COLUMN_HEAD, COLUMN_TAIL, TargetColumn

__all__ = [
    "get_csr_matrix",
    "marginal_score",
    "get_relation_similarity",
]


def get_csr_matrix(
    row_indices: numpy.ndarray,
    col_indices: numpy.ndarray,
    shape: Tuple[int, int],
) -> scipy.sparse.csr_matrix:
    """Create a sparse matrix, for the given non-zero locations."""
    # create sparse matrix of absolute counts
    matrix = scipy.sparse.coo_matrix(
        (numpy.ones(row_indices.shape, dtype=numpy.float32), (row_indices, col_indices)),
        shape=shape,
    ).tocsr()
    # normalize to relative counts
    return sklearn_normalize(matrix, norm="l1")


def marginal_score(
    entity_relation_batch: torch.LongTensor,
    per_entity: Optional[scipy.sparse.csr_matrix],
    per_relation: Optional[scipy.sparse.csr_matrix],
    num_entities: int,
) -> torch.FloatTensor:
    """Shared code for computing entity scores from marginals."""
    batch_size = entity_relation_batch.shape[0]

    # base case
    if per_entity is None and per_relation is None:
        return torch.full(size=(batch_size, num_entities), fill_value=1 / num_entities)

    e, r = entity_relation_batch.cpu().numpy().T

    if per_relation is not None and per_entity is None:
        scores = per_relation[r]
    elif per_relation is None and per_entity is not None:
        scores = per_entity[e]
    elif per_relation is not None and per_entity is not None:
        e_score = per_entity[e]
        r_score = per_relation[r]
        scores = e_score.multiply(r_score)
        scores = sklearn_normalize(scores, norm="l1", axis=1)
    else:
        raise AssertionError  # for mypy

    # note: we need to work with dense arrays only to comply with returning torch tensors. Otherwise, we could
    # stay sparse here, with a potential of a huge memory benefit on large datasets!
    return torch.from_numpy(scores.todense())


def sparsify(
    matrix: numpy.ndarray,
    threshold: Optional[float] = None,
) -> scipy.sparse.spmatrix:
    """
    Sparsify a matrix.

    :param matrix: shape: (m, n)
        the (dense) matrix
    :param threshold:
        the absolute threshold for sparsification

    :return: shape: (m, n)
        a sparsified matrix
    """
    if threshold is not None:
        matrix = numpy.copy(matrix)
        matrix[matrix < threshold] = 0.0
    sparse = scipy.sparse.csr_matrix(matrix)
    sparse.eliminate_zeros()
    return sparse


def entity_pair_matrix(
    triples_factory: CoreTriplesFactory,
    order: Tuple[TargetColumn, TargetColumn] = (COLUMN_HEAD, COLUMN_TAIL),
) -> scipy.sparse.spmatrix:
    """
    Create a sparse matrix of entity-pairs for each relation.

    :param triples:
        the triples factory
    :param order:
        the order of entities

    :return: shape: `(num_relations, num_entities ** 2)`
        a sparse matrix, where each row contains the one-hot encoded set of entity pairs
    """
    # comment: we cannot use leakage.mapped_triples_to_sparse_matrices, since we need to 
    #          retain the full column count, num_entities ** 2, for mapping back to entities 
    mapped_triples = numpy.asarray(triples_factory.mapped_triples)
    first, second = order
    return scipy.sparse.coo_matrix(
        (
            numpy.ones((mapped_triples.shape[0],), dtype=int),
            (
                mapped_triples[:, 1],
                triples_factory.num_entities * mapped_triples[:, first] + mapped_triples[:, second],
            ),
        ),
        shape=(triples_factory.num_relations, triples_factory.num_entities**2),
    ).tocsr()


def get_relation_similarity(
    triples_factory: CoreTriplesFactory,
    to_inverse: bool = False,
    threshold: Optional[float] = None,
) -> scipy.sparse.csr_matrix:
    """Get the relation similarity."""
    assert triples_factory.num_entities * triples_factory.num_relations < numpy.iinfo(int_type=int).max
    r = entity_pair_matrix(triples_factory=triples_factory, order=(COLUMN_HEAD, COLUMN_TAIL))
    r2 = entity_pair_matrix(triples_factory=triples_factory, order=(COLUMN_TAIL, COLUMN_HEAD)) if to_inverse else r
    return sparsify(jaccard_similarity_scipy(a=r, b=r2), threshold=threshold)
