# -*- coding: utf-8 -*-

import logging
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    'create_mapped_triples',
    'create_mappings',
]

log = logging.getLogger(__name__)


def create_mapped_triples(triples: np.ndarray,
                          entity_to_id: Optional[Dict[int, str]] = None,
                          rel_to_id: Optional[Dict[int, str]] = None) -> np.ndarray:
    """"""
    if entity_to_id is None or rel_to_id is None:
        entity_to_id, rel_to_id = create_mappings(triples)

    subject_column = np.vectorize(entity_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(triples[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order
    return np.unique(ar=triples_of_ids, axis=0), entity_to_id, rel_to_id


def create_mappings(triples: np.ndarray) -> Tuple[Dict[str, int], Dict[str, int]]:
    """"""
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())

    entity_to_id: Dict[int, str] = {
        value: key
        for key, value in enumerate(entities)
    }

    rel_to_id: Dict[int, str] = {
        value: key
        for key, value in enumerate(relations)
    }

    return entity_to_id, rel_to_id
