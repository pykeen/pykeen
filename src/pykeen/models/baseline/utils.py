# -*- coding: utf-8 -*-

"""Utilities for non-parametric baseline models."""

from typing import Optional, Tuple

import numpy
import scipy.sparse
import torch
from sklearn.preprocessing import normalize as sklearn_normalize

from ...triples import CoreTriplesFactory

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


def get_relation_similarity(
    triples_factory: CoreTriplesFactory,
    to_inverse: bool = False,
    threshold: Optional[float] = None,
) -> scipy.sparse.csr_matrix:
    """Get the relation similarity."""
    # TODO: overlap with inverse triple detection
    assert triples_factory.num_entities * triples_factory.num_relations < numpy.iinfo(int_type=int).max
    mapped_triples = numpy.asarray(triples_factory.mapped_triples)
    r = scipy.sparse.coo_matrix(
        (
            numpy.ones((mapped_triples.shape[0],), dtype=int),
            (
                mapped_triples[:, 1],
                triples_factory.num_entities * mapped_triples[:, 0] + mapped_triples[:, 2],
            ),
        ),
        shape=(triples_factory.num_relations, triples_factory.num_entities ** 2),
    )
    cardinality = numpy.asarray(r.sum(axis=1)).squeeze(axis=-1)
    if not to_inverse:
        return _help_get_relation_similarity(r, r, cardinality=cardinality, threshold=threshold)

    r2 = scipy.sparse.coo_matrix(
        (
            numpy.ones((mapped_triples.shape[0],), dtype=int),
            (
                mapped_triples[:, 1],
                triples_factory.num_entities * mapped_triples[:, 2] + mapped_triples[:, 0],
            ),
        ),
        shape=(triples_factory.num_relations, triples_factory.num_entities ** 2),
    )
    return _help_get_relation_similarity(r, r2, cardinality=cardinality, threshold=threshold)


def _help_get_relation_similarity(
    r,
    r2,
    cardinality,
    threshold: Optional[float] = None,
) -> scipy.sparse.csr_matrix:
    intersection = numpy.asarray((r @ r2.T).todense())
    union = cardinality[:, None] + cardinality[None, :] - intersection
    sim = intersection.astype(numpy.float32) / union.astype(numpy.float32)
    if threshold is not None:
        sim[sim < threshold] = 0.0
    sim = scipy.sparse.csr_matrix(sim)
    sim.eliminate_zeros()
    return sim
