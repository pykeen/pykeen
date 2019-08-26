# -*- coding: utf-8 -*-

"""Utilities for POEM."""

import logging
from typing import Union

import numpy
import torch

__all__ = [
    'l2_regularization',
    'resolve_device',
    'slice_triples',
    'slice_doubles',
]

log = logging.getLogger(__name__)


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


def resolve_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        log.info('No cuda devices were available. The model runs on CPU')
    return device


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
