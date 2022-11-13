# -*- coding: utf-8 -*-

"""Compute kernels for common sub-tasks."""

import torch

from ..utils import einsum

__all__ = [
    "batched_dot",
]


def _batched_dot_manual(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return (a * b).sum(dim=-1)


# TODO benchmark
def _batched_dot_matmul(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return (a.unsqueeze(dim=-2) @ b.unsqueeze(dim=-1)).view(a.shape[:-1])


# TODO benchmark
def _batched_dot_einsum(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return einsum("...i,...i->...", a, b)


def batched_dot(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute "element-wise" dot-product between batched vectors."""
    return _batched_dot_manual(a, b)
