# -*- coding: utf-8 -*-

"""Utilities for POEM."""

__all__ = [
    'slice_triples',
    'slice_doubles',
]


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    return (
        triples[:, 0:1],  # heads
        triples[:, 1:2],  # relations
        triples[:, 2:3],  # tails
    )


def slice_doubles(doubles):
    """Get the heads and relations from a matrix of doubles."""
    return (
        doubles[:, 0:1],  # heads
        doubles[:, 1:2],  # relations
    )
