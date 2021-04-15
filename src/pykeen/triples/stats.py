# -*- coding: utf-8 -*-

"""Compute statistics of a KG to be able to interpret the performance of KGE models."""

from collections import Counter
from typing import Mapping

import numpy as np


def compute_number_tails_per_head_relation_tuples(triples: np.ndarray) -> Mapping[str, int]:
    """Compute number of tails per head-relation pairs."""
    return _count_two_columns(triples, slice(0, 1), slice(1, 2))


def compute_number_heads_per_tail_relation_tuples(triples: np.ndarray) -> Mapping[str, int]:
    """Compute number of heads per relation-tail pairs."""
    return _count_two_columns(triples, slice(1, 2), slice(2, 3))


def _count_two_columns(triples: np.ndarray, c1_slice: slice, c2_slice: slice) -> Mapping[str, int]:
    """Compute number of heads per relation-tail pairs."""
    c1 = triples[:, c1_slice]
    c2 = triples[:, c2_slice]

    arr = np.concatenate([c1, c2], axis=-1).tolist()
    stats = Counter(map(tuple, arr))
    return _get_sorted_dict_from_counter(stats)


def _get_sorted_dict_from_counter(counter: Counter) -> Mapping[str, int]:
    """Return sorted dict for Counter tail."""
    return {
        f'{c1_label} {c2_label}': count
        for (c1_label, c2_label), count in counter.most_common()
    }
