# -*- coding: utf-8 -*-

"""Utilities for POEM."""

import logging
from typing import Iterable, List, Mapping, Optional, Type, TypeVar, Union

import numpy
import torch
from torch.optim import Adam, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

__all__ = [
    'l2_regularization',
    'resolve_device',
    'slice_triples',
    'slice_doubles',
    'split_list_in_batches_iter',
    'split_list_in_batches',
    'normalize_string',
    'get_cls',
    'optimizers',
    'get_optimizer_cls',
]

logger = logging.getLogger(__name__)


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
        logger.warning('No cuda devices were available. The model runs on CPU')
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


X = TypeVar('X')


def split_list_in_batches(input_list: List[X], batch_size: int) -> List[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return list(split_list_in_batches_iter(input_list=input_list, batch_size=batch_size))


def split_list_in_batches_iter(input_list: List[X], batch_size: int) -> Iterable[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return (
        input_list[i:i + batch_size]
        for i in range(0, len(input_list), batch_size)
    )


def normalize_string(s: str) -> str:
    """Normalize a string for lookup."""
    return s.lower().replace('-', '').replace('_', '')


def get_cls(
    query: Union[None, str, Type[X]],
    base: Type[X],
    lookup_dict: Mapping[str, Type[X]],
    default: Optional[Type[X]] = None,
) -> Type[X]:
    """Get a class by string, default, or implementation."""
    if query is None:
        if default is None:
            raise ValueError(f'No default {base.__name__} set')
        return default
    elif not isinstance(query, (str, type)):
        raise TypeError(f'Invalid {base.__name__} type: {type(query)} - {query}')
    elif isinstance(query, str):
        try:
            return lookup_dict[normalize_string(query)]
        except KeyError:
            raise ValueError(f'Invalid {base.__name__} name: {query}')
    elif issubclass(query, base):
        return query
    raise TypeError(f'Not subclass of {base.__name__}: {query}')


_OPTIMIZER_LIST = [
    Adam,
    SGD,
    AdamW,
    Adagrad,
    Adadelta,
    Adamax,
]
optimizers = {
    normalize_string(optimizer.__name__): optimizer
    for optimizer in _OPTIMIZER_LIST
}


def get_optimizer_cls(query: Union[None, str, Type[Optimizer]]) -> Type[Optimizer]:
    """Get the optimizer class."""
    return get_cls(
        query,
        base=Optimizer,
        lookup_dict=optimizers,
        default=Adagrad,
    )
