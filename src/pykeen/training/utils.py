# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

from typing import Callable, Iterable, TypeVar

import numpy

from ..utils import split_list_in_batches_iter

__all__ = [
    "lazy_compile_random_batches",
]

X = TypeVar("X")


def lazy_compile_random_batches(
    indices: numpy.ndarray,
    batch_size: int,
    batch_compiler: Callable[[numpy.ndarray], X],
) -> Iterable[X]:
    """Compile training batches of given size using random shuffling.

    :param indices:
        The indices to training samples. Is modified through shuffling.
    :param batch_size:
        The desired batch size.
    :param batch_compiler:
        A callable which takes the indices to put into a batch, and returns the batch of elements.
    """
    # Shuffle each epoch
    numpy.random.shuffle(indices)

    # Lazy-splitting into batches
    index_batches = split_list_in_batches_iter(indices, batch_size=batch_size)

    return map(batch_compiler, index_batches)
