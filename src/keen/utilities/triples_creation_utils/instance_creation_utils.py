# -*- coding: utf-8 -*-

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
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


# TODO: Make sure that no negative example is contained in positive set
def create_negative_triples(seed, pos_triples, filter_neg_triples=False):
    """

    :param seed:
    :param pos_triples:
    :param ratio_of_negative_triples:
    :return:
    """

    np.random.seed(seed=seed)

    num_pos_triples = pos_triples.shape[0]
    num_subj_corrupt = num_pos_triples // 2

    subjects = pos_triples[:, 0:1]
    objects = pos_triples[:, 2:3]
    relations = pos_triples[:, 1:2]
    permuted_subjects = np.random.permutation(subjects)
    permuted_objects = np.random.permutation(objects)

    triples_manp_subjs = np.concatenate(
        [permuted_subjects[:num_subj_corrupt, :], relations[:num_subj_corrupt, :], objects[:num_subj_corrupt, :]],
        axis=1)
    triples_manp_objs = np.concatenate(
        [subjects[num_subj_corrupt:, :], relations[num_subj_corrupt:, :], permuted_objects[num_subj_corrupt:, :]],
        axis=1)
    neg_triples = np.concatenate([triples_manp_subjs, triples_manp_objs], axis=0)

    if filter_neg_triples:
        filtered_neg_triples = np.setdiff1d(neg_triples, pos_triples)
        log.info("Filtered out %d " % (len(neg_triples) - len(filtered_neg_triples)))
        neg_triples = filtered_neg_triples
        print(filtered_neg_triples)
        exit(0)

    return neg_triples
