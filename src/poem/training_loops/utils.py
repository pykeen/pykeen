# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

from typing import List

__all__ = [
    'split_list_in_batches',
]


def split_list_in_batches(input_list, batch_size: int) -> List:
    """Split a list of instances in batches of size batch_size."""
    return [
        input_list[i:i + batch_size]
        for i in range(0, len(input_list), batch_size)
    ]
