# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

from typing import Iterable, List, Tuple, TypeVar

import numpy
import torch

__all__ = [
    'apply_label_smoothing',
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


def apply_label_smoothing(
        labels: torch.FloatTensor,
        epsilon: float,
        num_classes: int,
) -> torch.FloatTensor:
    """Apply label smoothing to a target tensor.

    Redistributes epsilon probability mass from the true target uniformly to the remaining classes by replacing
        * a hard one by (1 - epsilon)
        * a hard zero by epsilon / (num_classes - 1)

    :param labels:
        The one-hot label tensor.
    :param epsilon:
        The smoothing parameter. Determines how much probability should be transferred from the true class to the
        other classes.
    :param num_classes:
        The number of classes.

    ..seealso:
        http://www.deeplearningbook.org/contents/regularization.html, chapter 7.5.1
    """
    new_label_true = (1.0 - epsilon)
    new_label_false = epsilon / (num_classes - 1)
    labels = new_label_true * labels + new_label_false * (1.0 - labels)
    return labels


def lazy_compile_random_batches(
    indices: numpy.ndarray,
    input_array: numpy.ndarray,
    target_array: numpy.ndarray,
    batch_size: int,
) -> Iterable[Tuple[numpy.ndarray, numpy.ndarray]]:
    """Compile training batches of given size using random shuffling."""
    # Shuffle each epoch
    numpy.random.shuffle(indices)

    # Lazy-splitting into batches
    index_batches = split_list_in_batches_iter(indices, batch_size=batch_size)

    # Re-order according to indices
    def _compile_batch_from_indices(batch_indices):
        input_batch = input_array[batch_indices]
        target_batch = target_array[batch_indices]
        return input_batch, target_batch

    return map(_compile_batch_from_indices, index_batches)
