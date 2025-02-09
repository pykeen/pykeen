"""Utilities for neural network components."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import torch

from ..typing import FloatTensor, LongTensor, OneOrSequence
from ..utils import upgrade_to_sequence

__all__ = [
    "apply_optional_bn",
    "safe_diagonal",
    "adjacency_tensor_to_stacked_matrix",
    "use_horizontal_stacking",
    "ShapeError",
    # Caches
]

logger = logging.getLogger(__name__)


def iter_matrix_power(matrix: torch.Tensor, max_iter: int) -> Iterable[torch.Tensor]:
    """
    Iterate over matrix powers.

    :param matrix: shape: `(n, n)`
        the square matrix
    :param max_iter:
        the maximum number of iterations.

    :yields: increasing matrix powers
    """
    yield matrix
    a = matrix
    for _ in range(max_iter - 1):
        # if the sparsity becomes too low, convert to a dense matrix
        # note: this heuristic is based on the memory consumption,
        # for a sparse matrix, we store 3 values per nnz (row index, column index, value)
        # performance-wise, it likely makes sense to switch even earlier
        # `torch.sparse.mm` can also deal with dense 2nd argument
        if a.is_sparse and a._nnz() >= a.numel() // 4:
            a = a.to_dense()
        # note: torch.sparse.mm only works for COO matrices;
        #       @ only works for CSR matrices
        if matrix.is_sparse_csr:
            a = matrix @ a
        else:
            a = torch.sparse.mm(matrix, a)
        yield a


def safe_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extract diagonal from a potentially sparse matrix.

    .. note ::
        this is a work-around as long as :func:`torch.diagonal` does not work for sparse tensors

    :param matrix: shape: `(n, n)`
        the matrix

    :return: shape: `(n,)`
        the diagonal values.
    """
    if not matrix.is_sparse:
        return torch.diagonal(matrix)

    # convert to COO, if necessary
    if matrix.is_sparse_csr:
        matrix = matrix.to_sparse_coo()

    n = matrix.shape[0]
    # we need to use indices here, since there may be zero diagonal entries
    indices = matrix._indices()
    mask = indices[0] == indices[1]
    diagonal_values = matrix._values()[mask]
    diagonal_indices = indices[0][mask]

    return torch.zeros(n, device=matrix.device).scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)


def use_horizontal_stacking(
    input_dim: int,
    output_dim: int,
) -> bool:
    """
    Determine a stacking direction based on the input and output dimension.

    The vertical stacking approach is suitable for low dimensional input and high dimensional output,
    because the projection to low dimensions is done first. While the horizontal stacking approach is good
    for high dimensional input and low dimensional output as the projection to high dimension is done last.

    :param input_dim:
        the layer's input dimension
    :param output_dim:
        the layer's output dimension

    :return:
        whether to use horizontal (True) or vertical stacking

    .. seealso :: [thanapalasingam2021]_
    """
    return input_dim > output_dim


def adjacency_tensor_to_stacked_matrix(
    num_relations: int,
    num_entities: int,
    source: LongTensor,
    target: LongTensor,
    edge_type: LongTensor,
    edge_weights: FloatTensor | None = None,
    horizontal: bool = True,
) -> torch.Tensor:
    """
    Stack adjacency matrices as described in [thanapalasingam2021]_.

    This method re-arranges the (sparse) adjacency tensor of shape
    `(num_entities, num_relations, num_entities)` to a sparse adjacency matrix of shape
    `(num_entities, num_relations * num_entities)` (horizontal stacking) or
    `(num_entities * num_relations, num_entities)` (vertical stacking). Thereby, we can perform the relation-specific
    message passing of R-GCN by a single sparse matrix multiplication (and some additional pre- and/or
    post-processing) of the inputs.

    :param num_relations:
        the number of relations
    :param num_entities:
        the number of entities
    :param source: shape: (num_triples,)
        the source entity indices
    :param target: shape: (num_triples,)
        the target entity indices
    :param edge_type: shape: (num_triples,)
        the edge type, i.e., relation ID
    :param edge_weights: shape: (num_triples,)
        scalar edge weights
    :param horizontal:
        whether to use horizontal or vertical stacking

    :return: shape: `(num_entities * num_relations, num_entities)` or `(num_entities, num_entities * num_relations)`
        the stacked adjacency matrix
    """
    offset = edge_type * num_entities
    if horizontal:
        size = (num_entities, num_relations * num_entities)
        target = offset + target
    else:
        size = (num_relations * num_entities, num_entities)
        source = offset + source
    indices = torch.stack([source, target], dim=0)
    if edge_weights is None:
        edge_weights = torch.ones_like(source, dtype=torch.get_default_dtype())
    return torch.sparse_coo_tensor(
        indices=indices,
        values=edge_weights,
        size=size,
    )


class ShapeError(ValueError):
    """An error for a mismatch in shapes."""

    def __init__(self, shape: Sequence[int], reference: Sequence[int]) -> None:
        """
        Initialize the error.

        :param shape: the mismatching shape
        :param reference: the expected shape
        """
        super().__init__(f"shape {shape} does not match expected shape {reference}")

    @classmethod
    def verify(cls, shape: OneOrSequence[int], reference: OneOrSequence[int] | None) -> Sequence[int]:
        """
        Raise an exception if the shape does not match the reference.

        This method normalizes the shapes first.

        :param shape:
            the shape to check
        :param reference:
            the reference shape. If None, the shape always matches.

        :raises ShapeError:
            if the two shapes do not match.

        :return:
            the normalized shape
        """
        shape = upgrade_to_sequence(shape)
        if reference is None:
            return shape
        reference = upgrade_to_sequence(reference)
        if reference != shape:
            # darglint does not like
            # raise cls(shape=shape, reference=reference)
            raise ShapeError(shape=shape, reference=reference)
        return shape


def apply_optional_bn(x: FloatTensor, batch_norm: torch.nn.BatchNorm1d | None = None) -> FloatTensor:
    """Apply optional batch normalization.

    Supports multiple batch dimensions.

    :param x: shape: ``(..., d)```
        The input tensor.
    :param batch_norm:
        An optional batch normalization layer.

    :return: shape: ``(..., d)```
        The normalized tensor.
    """
    if batch_norm is None:
        return x
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    x = batch_norm(x)
    return x.view(*shape)
