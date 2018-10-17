# -*- coding: utf-8 -*-

"""Script to compute mean rank and hits@k."""

import logging
import timeit

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _hash_triples(triples):
    """

    :param triples:
    :return:
    """
    return hash(tuple(triples))


def _compute_hits_at_k(hits_at_k_dict, rank_of_positive_subject_based, rank_of_positive_object_based):
    """

    :param hits_at_k_dict:
    :param rank_of_positive_subject_based:
    :param rank_of_positive_object_based:
    :return:
    """
    for k, value in hits_at_k_dict.items():
        if rank_of_positive_subject_based < k:
            value.append(1.)
        else:
            value.append(0.)

        if rank_of_positive_object_based < k:
            value.append(1.)
        else:
            value.append(0.)

    return hits_at_k_dict


def _create_corrupted_triples(triple, all_entities, device):
    """

    :param triple:
    :param all_entities:
    :param device:
    :return:
    """
    candidate_entities_subject_based = np.delete(arr=all_entities, obj=triple[0:1])
    candidate_entities_subject_based = np.reshape(candidate_entities_subject_based, newshape=(-1, 1))
    candidate_entities_object_based = np.delete(arr=all_entities, obj=triple[2:3])
    candidate_entities_object_based = np.reshape(candidate_entities_object_based, newshape=(-1, 1))

    # Extract current test tuple: Either (subject,predicate) or (predicate,object)
    tuple_subject_based = np.reshape(a=triple[1:3], newshape=(1, 2))
    tuple_object_based = np.reshape(a=triple[0:2], newshape=(1, 2))

    # Copy current test tuple
    tuples_subject_based = np.repeat(a=tuple_subject_based, repeats=candidate_entities_subject_based.shape[0],
                                     axis=0)
    tuples_object_based = np.repeat(a=tuple_object_based, repeats=candidate_entities_object_based.shape[0], axis=0)

    corrupted_subject_based = np.concatenate([candidate_entities_subject_based, tuples_subject_based], axis=1)
    corrupted_subject_based = torch.tensor(corrupted_subject_based, dtype=torch.long, device=device)

    corrupted_object_based = np.concatenate([tuples_object_based, candidate_entities_object_based], axis=1)
    corrupted_object_based = torch.tensor(corrupted_object_based, dtype=torch.long, device=device)

    return corrupted_subject_based, corrupted_object_based


def _filter_corrupted_triples(corrupted_subject_based, corrupted_object_based, all_pos_triples_hashed):
    """

    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param all_pos_triples_hashed:
    :return:
    """
    # TODO: Check
    corrupted_subject_based_hashed = np.apply_along_axis(_hash_triples, 1, corrupted_subject_based)
    mask = np.in1d(corrupted_subject_based_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]
    corrupted_subject_based = corrupted_subject_based[mask]

    corrupted_object_based_hashed = np.apply_along_axis(_hash_triples, 1, corrupted_object_based)
    mask = np.in1d(corrupted_object_based_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]

    if mask.size == 0:
        raise Exception("User selected filtered metric computation, but all corrupted triples exists"
                        "also a positive triples.")
    corrupted_object_based = corrupted_object_based[mask]

    return corrupted_subject_based, corrupted_object_based


def _compute_filtered_rank(kg_embedding_model, pos_triple, corrupted_subject_based, corrupted_object_based, device,
                           all_pos_triples_hashed):
    """

    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed:
    :return:
    """

    corrupted_subject_based, corrupted_object_based = _filter_corrupted_triples(
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        all_pos_triples_hashed=all_pos_triples_hashed)

    rank_of_positive_subject_based, rank_of_positive_object_based = _compute_rank(kg_embedding_model=kg_embedding_model,
                                                                                  pos_triple=pos_triple,
                                                                                  corrupted_subject_based=corrupted_subject_based,
                                                                                  corrupted_object_based=corrupted_object_based,
                                                                                  device=device)

    return rank_of_positive_subject_based, rank_of_positive_object_based


def _compute_rank(kg_embedding_model, pos_triple, corrupted_subject_based, corrupted_object_based, device,
                  all_pos_triples_hashed=None):
    """

    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed:
    :return:
    """
    scores_of_corrupted_subjects = kg_embedding_model.predict(corrupted_subject_based)
    scores_of_corrupted_objects = kg_embedding_model.predict(corrupted_object_based)

    pos_triple = np.array(pos_triple)
    pos_triple = np.expand_dims(a=pos_triple, axis=0)
    pos_triple = torch.tensor(pos_triple, dtype=torch.long, device=device)

    score_of_positive = kg_embedding_model.predict(pos_triple)

    scores_subject_based = np.append(arr=scores_of_corrupted_subjects, values=score_of_positive)
    indice_of_pos_subject_based = scores_subject_based.size - 1

    scores_object_based = np.append(arr=scores_of_corrupted_objects, values=score_of_positive)
    indice_of_pos_object_based = scores_object_based.size - 1

    _, sorted_score_indices_subject_based = torch.sort(torch.tensor(scores_subject_based, dtype=torch.float),
                                                       descending=False)
    sorted_score_indices_subject_based = sorted_score_indices_subject_based.cpu().numpy()

    _, sorted_score_indices_object_based = torch.sort(torch.tensor(scores_object_based, dtype=torch.float),
                                                      descending=False)
    sorted_score_indices_object_based = sorted_score_indices_object_based.cpu().numpy()

    # Get index of first occurence that fulfills the condition
    rank_of_positive_subject_based = np.where(sorted_score_indices_subject_based == indice_of_pos_subject_based)[0][0]
    rank_of_positive_object_based = np.where(sorted_score_indices_object_based == indice_of_pos_object_based)[0][0]

    return rank_of_positive_subject_based, rank_of_positive_object_based


def compute_metrics(all_entities, kg_embedding_model, mapped_train_triples, mapped_test_triples, device,
                    filter_neg_triples=False):
    """

    :param all_entities:
    :param kg_embedding_model:
    :param mapped_train_triples:
    :param mapped_test_triples:
    :param device:
    :param filter_neg_triples:
    :return:
    """
    start = timeit.default_timer()
    ranks = []
    hits_at_k_dict = {k: [] for k in [1, 3, 5, 10]}
    kg_embedding_model = kg_embedding_model.to(device)
    all_pos_triples = np.concatenate([mapped_train_triples, mapped_test_triples], axis=0)
    all_pos_triples_hashed = np.apply_along_axis(_hash_triples, 1, all_pos_triples)

    compute_rank_fct = _compute_filtered_rank if filter_neg_triples else _compute_rank

    # Corrupt triples
    for triple_nmbr, pos_triple in enumerate(mapped_test_triples):
        corrupted_subject_based, corrupted_object_based = _create_corrupted_triples(triple=pos_triple,
                                                                                    all_entities=all_entities,
                                                                                    device=device)

        rank_of_positive_subject_based, rank_of_positive_object_based = compute_rank_fct(
            kg_embedding_model=kg_embedding_model,
            pos_triple=pos_triple,
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            device=device,
            all_pos_triples_hashed=all_pos_triples_hashed)

        ranks.append(rank_of_positive_subject_based)

        ranks.append(rank_of_positive_object_based)

        # Compute hits@k for k in {1,3,5,10}
        hits_at_k_dict.update(
            _compute_hits_at_k(hits_at_k_dict, rank_of_positive_subject_based=rank_of_positive_subject_based,
                               rank_of_positive_object_based=rank_of_positive_object_based))

    mean_rank = np.mean(ranks)

    for k, value in hits_at_k_dict.items():
        hits_at_k_dict[k] = np.mean(value)

    stop = timeit.default_timer()
    log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))

    return mean_rank, hits_at_k_dict
