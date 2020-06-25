# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

from typing import Callable, Iterable, TypeVar

import numpy
import torch

from ..utils import split_list_in_batches_iter

__all__ = [
    'apply_label_smoothing',
    'lazy_compile_random_batches',
]

X = TypeVar('X')


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
