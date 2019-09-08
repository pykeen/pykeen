# -*- coding: utf-8 -*-

"""Compute statistics of a KG to be able to interpret the performance of KGE models."""

from collections import Counter
from typing import Mapping

import numpy as np


def compute_number_objects_per_subject_relation_tuples(triples: np.ndarray) -> Mapping[str, int]:
    """Compute number of objects subject-relation pairs."""
    return _count_two_columns(triples, slice(0, 1), slice(1, 2))


def compute_number_subjects_per_object_relation_tuples(triples: np.ndarray) -> Mapping[str, int]:
    """Compute number of subjects per relation-object pairs."""
    return _count_two_columns(triples, slice(1, 2), slice(2, 3))


def _count_two_columns(triples: np.ndarray, c1_slice, c2_slice):
    """Compute number of subjects per relation-object pairs."""
    c1 = triples[:, c1_slice]
    c2 = triples[:, c2_slice]

    arr = np.concatenate([c1, c2], axis=-1).tolist()
    stats = Counter(map(tuple, arr))
    stats = _get_sorted_dict_from_counter(stats)
    return stats


def _get_sorted_dict_from_counter(counter: Counter) -> Mapping[str, int]:
    """Return sorted dict for Counter object."""
    return {
        f'{c1_label} {c2_label}': count
        for (c1_label, c2_label), count in counter.most_common()
    }
