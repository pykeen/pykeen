# -*- coding: utf-8 -*-

"""Utilities for POEM."""

import numpy
import torch

__all__ = [
    'l2_regularization',
    'slice_triples',
    'slice_doubles',
]


def l2_regularization(
        *xs: torch.Tensor,
        normalize: bool = False
) -> torch.Tensor:
    """
    Compute squared L2-regularization term.

    :param xs: a list of torch.Tensor
        The tensors for which to compute the regularization.
    :param normalize:
        Whether to divide the term by the total number of elements in the tensors.

    :return: The sum of squared value across all tensors.
    """
    regularization_term = sum(x.pow(2).sum() for x in xs)

    # Normalize by the number of elements in the tensors for dimensionality-independent weight tuning.
    if normalize:
        regularization_term /= sum(numpy.prod(x.shape) for x in xs)

    return regularization_term


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
