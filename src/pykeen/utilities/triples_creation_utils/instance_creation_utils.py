# -*- coding: utf-8 -*-

import logging

import numpy as np

log = logging.getLogger(__name__)


def create_mapped_triples(triples, entity_to_id=None, rel_to_id=None):
    """

    :param path_to_kg:
    :return:
    """
    if entity_to_id is None or rel_to_id is None:
        entity_to_id, rel_to_id = create_mappings(triples)

    subject_column = np.vectorize(entity_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(triples[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    triples_of_ids = np.unique(ar=triples_of_ids, axis=0)

    return triples_of_ids, entity_to_id, rel_to_id


def create_mappings(triples):
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())
    entity_to_id = {value: key for key, value in enumerate(entities)}
    rel_to_id = {value: key for key, value in enumerate(relations)}

    return entity_to_id, rel_to_id
