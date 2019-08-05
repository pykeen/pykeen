# -*- coding: utf-8 -*-

"""Utilities for encoding sentences."""

from typing import Dict

import numpy as np


def load_entity_to_descriptions(path_to_file: str) -> np.array:
    """Load entity to descriptions file."""
    return np.loadtxt(
        fname=path_to_file,
        dtype=str,
        comments='@Comment@ Subject Predicate Object',
        delimiter='\t',
    )


def create_entity_to_desciption_mappings(entity_description_matrix: np.array) -> Dict[str, str]:
    """Map entities to descriptions."""
    entities = entity_description_matrix[:, 0]
    descriptions = entity_description_matrix[:, 1]

    # TODO compress into dictionary comprehension
    entity_to_desciption = {}

    for entity, desc in zip(entities, descriptions):
        # Check whether entity/description is empty or None
        if not entity or entity is None or not desc or desc is None:
            continue

        entity_to_desciption[entity] = desc

    return entity_to_desciption
