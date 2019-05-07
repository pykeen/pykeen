# -*- coding: utf-8 -*-

"""Utils to process numerical literals."""

import numpy as np
from typing import Dict
import logging

log = logging.getLogger(__name__)


def create_matix_of_literals(numeric_triples: np.array, entity_to_id: Dict) -> np.ndarray:
    """Create matrix of literals where each row corresponds to an entity and each column to a literal."""
    data_relations = np.unique(np.ndarray.flatten(numeric_triples[:, 1:2]))
    data_rel_to_id: Dict[str, int] = {
        value: key
        for key, value in enumerate(data_relations)
    }
    # Prepare literal matrix, set every literal to zero, and afterwards fill in the corresponding value if available
    num_literals = np.zeros([len(entity_to_id), len(data_rel_to_id)], dtype=np.float32)

    # ToDo: Vecotrize code
    for i, (h, r, lit) in enumerate(numeric_triples):
        try:
            # row define entity, and column the literal. Set the corresponding literal for the entity
            num_literals[entity_to_id[h], data_rel_to_id[r]] = lit
        except KeyError:
            log.info("Either entity or relation to literal doesn't exist.")
            continue

    return num_literals
