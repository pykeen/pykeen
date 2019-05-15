# -*- coding: utf-8 -*-

"""Utilities for POEM."""

from typing import Iterable

from torch import Tensor, nn

__all__ = [
    'slice_triples',
    'get_params',
]


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    return (
        triples[:, 0:1],  # heads
        triples[:, 1:2],  # relations
        triples[:, 2:3],  # tails
    )


def get_params(module: nn.Module) -> Iterable[Tensor]:
    return filter(lambda p: p.requires_grad, module.parameters())
