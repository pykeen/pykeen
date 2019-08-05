# -*- coding: utf-8 -*-

"""Instance creation utilities."""

import logging
from collections import defaultdict
from typing import Dict

import numpy as np
from tqdm import tqdm

__all__ = [
    'create_multi_label_instances',
    'create_multi_label_objects_instance',
    'create_multi_label_relation_instances',
    'create_matrix_of_literals'
]

log = logging.getLogger(__name__)


def create_multi_label_relation_instances(
        triples: np.array,
        create_class_other=False,
) -> Dict[tuple, np.array]:
    """Create for each (s,o) pair the multi relation label."""
    log.info(f'Creating multi label relations instance')

    s_t_to_multi_relations = create_multi_label_instances(
        triples,
        element_1_index=0,
        element_2_index=2,
        label_index=1,
    )
    log.info(f'Created multi label relations instance')

    return s_t_to_multi_relations


def create_multi_label_objects_instance(triples: np.array) -> Dict[tuple, np.array]:
    """Create for each (s,r) pair the multi object label."""
    log.info(f'Creating multi label objects instance')

    s_r_to_multi_objects_new = create_multi_label_instances(
        triples,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
    )

    log.info(f'Created multi label objects instance')

    return s_r_to_multi_objects_new


def create_multi_label_instances(
        triples: np.array,
        element_1_index: int,
        element_2_index: int,
        label_index: int,
) -> Dict[tuple, np.array]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)
    for row in tqdm(triples):
        instance_to_multi_label[(row[element_1_index], row[element_2_index])].add(row[label_index])

    # Create lists out of sets for proper numpy indexing when loading the labels
    instance_to_multi_label_new = {
        key: list(value)
        for key, value in instance_to_multi_label.items()
    }

    return instance_to_multi_label_new


def create_matrix_of_literals(numeric_triples: np.array, entity_to_id: Dict) -> np.ndarray:
    """Create matrix of literals where each row corresponds to an entity and each column to a literal."""
    data_relations = np.unique(np.ndarray.flatten(numeric_triples[:, 1:2]))
    data_rel_to_id: Dict[str, int] = {
        value: key
        for key, value in enumerate(data_relations)
    }
    # Prepare literal matrix, set every literal to zero, and afterwards fill in the corresponding value if available
    num_literals = np.zeros([len(entity_to_id), len(data_rel_to_id)], dtype=np.float32)

    # TODO vectorize code
    for i, (h, r, lit) in enumerate(numeric_triples):
        try:
            # row define entity, and column the literal. Set the corresponding literal for the entity
            num_literals[entity_to_id[h], data_rel_to_id[r]] = lit
        except KeyError:
            log.info("Either entity or relation to literal doesn't exist.")
            continue

    return num_literals, data_rel_to_id
