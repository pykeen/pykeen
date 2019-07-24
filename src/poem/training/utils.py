# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

from typing import Iterable, List, TypeVar

__all__ = [
    'split_list_in_batches_iter',
    'split_list_in_batches',
]

X = TypeVar('X')


def split_list_in_batches_iter(input_list: List[X], batch_size: int) -> Iterable[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return (
        input_list[i:i + batch_size]
        for i in range(0, len(input_list), batch_size)
    )


def split_list_in_batches(input_list: List[X], batch_size: int) -> List[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return list(split_list_in_batches_iter(input_list=input_list, batch_size=batch_size))
