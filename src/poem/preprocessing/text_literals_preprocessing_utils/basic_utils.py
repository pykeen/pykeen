# -*- coding: utf-8 -*-


import numpy as np
from typing import Dict


def load_entity_to_descriptions(path_to_file: str) -> np.array:
    """Load entity to descriptions file."""

    entity_description_matrix = np.loadtxt(
        fname=path_to_file,
        dtype=str,
        comments='@Comment@ Subject Predicate Object',
        delimiter='\t',
    )

    return entity_description_matrix


def create_entity_to_desciption_mappings(entity_description_matrix: np.array) -> Dict[str, str]:
    """Map entities to descriptions"""

    entities = entity_description_matrix[:, 0]
    descriptions = entity_description_matrix[:, 1]

    entity_to_desciption = {}

    for i in range(len(entities)):
        entity = entities[i]
        desc = descriptions[i]

        # Check whether entity/description is empty or None
        if not entity or entity is None or not desc or desc is None:
            continue

        entity_to_desciption[entity] = desc

    return entity_to_desciption
