# -*- coding: utf-8 -*-

"""Compute statistics of a KG to be able to interpret the performance of KGE models."""

import operator
from collections import Counter, OrderedDict

import numpy as np


def get_sorted_dict_from_counter(counter):
    """Return sorted dict for Counter object."""
    temp_dict = OrderedDict()

    for key, value in counter.items():
        temp_dict[' '.join(key)] = value

    sorted_x = sorted(temp_dict.items(), key=operator.itemgetter(1))
    sorted_x = OrderedDict(sorted_x)

    return sorted_x


def compute_number_objects_per_subject_relation_tuples(triples):
    """Compute number of objects subject-relation pairs."""
    subjects = triples[:, 0:1]
    relations = triples[:, 1:2]
    subjs_rels = np.concatenate([subjects, relations], axis=-1).tolist()
    temp_tuple = map(tuple, subjs_rels)
    stats = Counter(temp_tuple)

    stats = get_sorted_dict_from_counter(stats)

    return stats


def compute_number_subjects_per_object_relation_tuples(triples):
    """Compute number of subjects per relation-object pairs."""
    objects = triples[:, 2:3]
    relations = triples[:, 1:2]
    objs_rels = np.concatenate([relations, objects], axis=-1).tolist()
    temp_tuple = map(tuple, objs_rels)
    stats = Counter(temp_tuple)

    stats = get_sorted_dict_from_counter(stats)

    return stats
