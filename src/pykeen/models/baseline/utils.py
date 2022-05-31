# -*- coding: utf-8 -*-

"""Utilities for non-parametric baseline models."""

from typing import Optional, Tuple

import numpy
import scipy.sparse
import torch
from sklearn.preprocessing import normalize as sklearn_normalize

from ...triples import CoreTriplesFactory
from ...triples.leakage import jaccard_similarity_scipy, triples_factory_to_sparse_matrices

__all__ = [
    "get_csr_matrix",
    "marginal_score",
    "get_relation_similarity",
]


def get_csr_matrix(
    row_indices: numpy.ndarray,
    col_indices: numpy.ndarray,
    shape: Tuple[int, int],
    dtype: numpy.dtype = numpy.float32,
    norm: Optional[str] = "l1",
) -> scipy.sparse.csr_matrix:
    """
    Create a sparse matrix, with ones for the given non-zero locations.

    :param row_indices: shape: (nnz,)
        the non-zero row indices
    :param col_indices: shape: (nnz,)
        the non-zero column indices
    :param shape:
        the matrix' shape
    :param dtype:
        the data type to use
    :param norm:
        if not None, perform row-wise normalization with :func:`sklearn.preprocessing.normalize`

    :return: shape: shape
        a sparse csr matrix
    """
    # create sparse matrix of absolute counts
    matrix = scipy.sparse.coo_matrix(
        (numpy.ones(row_indices.shape, dtype=dtype), (row_indices, col_indices)),
        shape=shape,
    ).tocsr()
    if not norm:
        return matrix
    # normalize to relative counts
    return sklearn_normalize(matrix, norm=norm)


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


def get_relation_similarity(
    triples_factory: CoreTriplesFactory,
    threshold: Optional[float] = None,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Compute Jaccard similarity of relations' (and their inverse's) entity-pair sets.

    :param triples_factory:
        the triples factory
    :param threshold:
        an absolute sparsification threshold.

    :return: shape: (num_relations, num_relations)
        a pair of similarity matrices.
    """
    r, r_inv = triples_factory_to_sparse_matrices(triples_factory=triples_factory)
    sim, sim_inv = [sparsify(jaccard_similarity_scipy(r, r2), threshold=threshold) for r2 in (r, r_inv)]
    return sim, sim_inv
