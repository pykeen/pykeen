import numpy as np

from corpus_reader.walking_rdf_and_owl_reader import WROCReader
from evaluation_methods.mean_rank_evaluator import MeanRankEvaluator
from kg_embeddings_model.trans_e import TransE
from kg_embeddings_model.trans_h import TransH
from utilities.constants import CLASS_NAME, WROC_READER, TRANS_E, TRANS_H, MEAN_RANK_EVALUATOR


def get_evaluator(config):
    class_name = config[CLASS_NAME]

    if class_name == MEAN_RANK_EVALUATOR:
        return MeanRankEvaluator()


def get_reader(config):
    """

    :param config:
    :return:
    """
    class_name = config[CLASS_NAME]

    if class_name == WROC_READER:
        return WROCReader()


def get_kg_embedding_model(config):
    """

    :param config:
    :return:
    """
    class_name = config[CLASS_NAME]

    if class_name == TRANS_E:
        return TransE(config=config)
    elif class_name == TRANS_H:
        return TransH(config=config)


def create_triples_and_mappings(path_to_kg):
    """

    :param path_to_kg:
    :return:
    """
    data = np.loadtxt(fname=path_to_kg, dtype=str, comments='@Comment@ Subject Predicate Object')
    entities = list(set(np.ndarray.flatten(np.concatenate([data[:, 0:1], data[:, 2:3]])).tolist()))
    relations = list(set(np.ndarray.flatten(data[:, 1:2]).tolist()))
    entity_to_id = {value: key for key, value in enumerate(entities)}
    rel_to_id = {value: key for key, value in enumerate(relations)}

    subject_column = np.vectorize(entity_to_id.get)(data[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(data[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(data[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    triples_of_ids = np.unique(ar=triples_of_ids,axis=0)

    return triples_of_ids, entity_to_id, rel_to_id


# TODO: Make sure that no negative example is contained in positive set
def create_negative_triples(seed, pos_triples, ratio_of_negative_triples=None):
    """

    :param seed:
    :param pos_triples:
    :param ratio_of_negative_triples:
    :return:
    """

    np.random.seed(seed=seed)

    assert (ratio_of_negative_triples is not None)

    num_pos_triples = pos_triples.shape[0]

    subjects = pos_triples[:, 0:1]
    objects = pos_triples[:, 2:3]
    relations = pos_triples[:, 1:2]
    permuted_subjects = np.random.permutation(subjects)
    permuted_objects = np.random.permutation(objects)

    triples_manp_subjs = np.concatenate([permuted_subjects, relations, objects], axis=1)
    triples_manp_objs = np.concatenate([permuted_objects, relations, objects], axis=1)
    manipulated_triples = np.random.permutation(np.concatenate([triples_manp_subjs, triples_manp_objs], axis=0))

    if ratio_of_negative_triples != None:
        num_neg_triples = int(num_pos_triples * ratio_of_negative_triples)

    neg_triples = manipulated_triples[:num_neg_triples, :]
    neg_triples = np.unique(ar=neg_triples,axis=0)

    return neg_triples
