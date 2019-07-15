# -*- coding: utf-8 -*-

import logging
from collections import defaultdict
from typing import Dict

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def create_multi_label_relation_instances(
        triples: np.array,
        create_class_other=False
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
