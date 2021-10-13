# -*- coding: utf-8 -*-

"""Utilities for non-parametric baseline models."""

from typing import Optional, Tuple

import numpy
import scipy.sparse
import torch
from sklearn.preprocessing import normalize as sklearn_normalize

__all__ = [
    'get_csr_matrix',
    'marginal_score',
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
