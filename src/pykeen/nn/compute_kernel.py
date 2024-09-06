"""Compute kernels for common sub-tasks."""

from ..typing import FloatTensor
from ..utils import einsum

__all__ = [
    "batched_dot",
]


def _batched_dot_manual(
    a: FloatTensor,
    b: FloatTensor,
) -> FloatTensor:
    return (a * b).sum(dim=-1)


# TODO benchmark
def _batched_dot_matmul(
    a: FloatTensor,
    b: FloatTensor,
) -> FloatTensor:
    return (a.unsqueeze(dim=-2) @ b.unsqueeze(dim=-1)).view(a.shape[:-1])


# TODO benchmark
def _batched_dot_einsum(
    a: FloatTensor,
    b: FloatTensor,
) -> FloatTensor:
    return einsum("...i,...i->...", a, b)


def batched_dot(
    a: FloatTensor,
    b: FloatTensor,
) -> FloatTensor:
    """Compute "element-wise" dot-product between batched vectors."""
    return _batched_dot_manual(a, b)
