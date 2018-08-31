# -*- coding: utf-8 -*-
import logging
import timeit
import torch
import numpy as np

from utilities.evaluation_utils.evaluation_helper import get_stratey_for_corrupting

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def compute_mean_rank_and_hits_at_k(all_entities, kg_embedding_model, triples, device, k=10):
    start = timeit.default_timer()
    ranks_subject_based, hits_at_k_subject_based = _compute_metrics(all_entities=all_entities,
                                                                    kg_embedding_model=kg_embedding_model,
                                                                    triples=triples, corrupt_suject=True,device=device, k=k)

    ranks_object_based, hits_at_k_object_based = _compute_metrics(all_entities=all_entities,
                                                                  kg_embedding_model=kg_embedding_model,
                                                                  triples=triples, corrupt_suject=False, device=device,k=k)
    mean_rank = np.mean(ranks_subject_based + ranks_object_based)

    all_hits = hits_at_k_subject_based + hits_at_k_object_based
    num_of_candidate_triples = 2 * all_entities.shape[0]
    hits_at_k = np.sum(all_hits) / (num_of_candidate_triples)

    stop = timeit.default_timer()
    log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))

    return mean_rank, hits_at_k


def compute_mean_rank(all_entities, kg_embedding_model, triples, device):
    start = timeit.default_timer()
    ranks_subject_based, _ = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
                                              triples=triples, corrupt_suject=True, device=device)

    ranks_object_based, _ = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
                                             triples=triples, corrupt_suject=False, device=device)
    ranks = ranks_subject_based + ranks_object_based
    mean_rank = np.mean(ranks)
    log.info("Ranks in compute: %s" % ranks)
    log.info("MEan rank in compute: %s" % mean_rank)

    stop = timeit.default_timer()
    log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))

    return mean_rank


def compute_hits_at_k(all_entities, kg_embedding_model, triples, device, k=10):
    start = timeit.default_timer()
    _, hits_at_k_subject_based = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
                                                  triples=triples, corrupt_suject=True, k=k)

    _, hits_at_k_object_based = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
                                                 triples=triples, corrupt_suject=False, k=k)

    all_hits = hits_at_k_subject_based + hits_at_k_object_based
    num_of_candidate_triples = 2 * all_entities.shape[0]
    hits_at_k = np.sum(all_hits) / (num_of_candidate_triples)

    stop = timeit.default_timer()
    log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))

    return hits_at_k


def _compute_metrics(all_entities, kg_embedding_model, triples, corrupt_suject, device, k=10):
    ranks = []
    in_top_k = []


    column_to_maintain_offsets, corrupted_column_offsets, concatenate_fct = get_stratey_for_corrupting(
        corrupt_suject=corrupt_suject)

    start_of_columns_to_maintain, end_of_columns_to_maintain = column_to_maintain_offsets

    # Corrupt triples
    for row_nmbr, row in enumerate(triples):
        candidate_entities = np.delete(arr=all_entities,
                                       obj=row[start_of_columns_to_maintain:start_of_columns_to_maintain + 1])
        # Extract current test tuple: Either (subject,predicate) or (predicate,object)
        tuple = np.reshape(a=triples[row_nmbr, start_of_columns_to_maintain:end_of_columns_to_maintain], newshape=(1, 2))
        # Copy current test tuple
        tuples = np.repeat(a=tuple, repeats=candidate_entities.shape[0], axis=0)

        corrupted = concatenate_fct(candidate_entities=candidate_entities, tuples=tuples)
        scores_of_corrupted = kg_embedding_model.predict(corrupted)
        pos_triple = np.array(triples[row_nmbr])
        pos_triple = np.expand_dims(a=pos_triple, axis=0)
        pos_triple = torch.tensor(pos_triple,dtype=torch.long,device=device)

        score_of_positive = kg_embedding_model.predict(pos_triple)

        scores = np.append(arr=scores_of_corrupted, values=score_of_positive)
        indice_of_pos = scores.size - 1

        scores = np.argsort(a=scores)

        # Get index of first occurence that fulfills the condition
        ranks.append(np.where(scores == indice_of_pos)[0][0])
        log.info("Ranks: %s" % ranks)

        # print(scores)
        top_k = scores[-k:]


        if pos_triple in top_k:
            in_top_k.append(1.)

    return ranks, in_top_k
