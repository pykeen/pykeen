# -*- coding: utf-8 -*-
import logging
import timeit

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# def compute_mean_rank_and_hits_at_k(all_entities, kg_embedding_model, triples, device, k=10):
#     start = timeit.default_timer()
#     mean_rank, hits_at_k = _compute_metrics_new(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
#                                                 triples=triples, device=device)
#
#     # ranks_subject_based, hits_at_k_subject_based = _compute_metrics(all_entities=all_entities,
#     #                                                                 kg_embedding_model=kg_embedding_model,
#     #                                                                 triples=triples, corrupt_suject=True, device=device,
#     #                                                                 k=k)
#     #
#     # ranks_object_based, hits_at_k_object_based = _compute_metrics(all_entities=all_entities,
#     #                                                               kg_embedding_model=kg_embedding_model,
#     #                                                               triples=triples, corrupt_suject=False, device=device,
#     #                                                               k=k)
#     # mean_rank = np.mean(ranks_subject_based + ranks_object_based)
#     #
#     # all_hits = hits_at_k_subject_based + hits_at_k_object_based
#     # hits_at_k = all_hits / (2. * triples.size)
#     #
#     # print("hits_at_k_subject_based: ", hits_at_k_subject_based / triples.size)
#     # print("hits_at_k_object_based: ", hits_at_k_object_based / triples.size)
#
#     stop = timeit.default_timer()
#     log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))
#
#     return mean_rank, hits_at_k
#
#
# def compute_mean_rank(all_entities, kg_embedding_model, triples, device):
#     start = timeit.default_timer()
#     ranks_subject_based, _ = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
#                                               triples=triples, corrupt_suject=True, device=device)
#
#     ranks_object_based, _ = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
#                                              triples=triples, corrupt_suject=False, device=device)
#     ranks = ranks_subject_based + ranks_object_based
#     mean_rank = np.mean(ranks)
#
#     stop = timeit.default_timer()
#     log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))
#
#     return mean_rank
#
#
# def compute_hits_at_k(all_entities, kg_embedding_model, triples, device, k=10):
#
#     _, hits_at_k_subject_based = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
#                                                   triples=triples, corrupt_suject=True, k=k)
#
#     _, hits_at_k_object_based = _compute_metrics(all_entities=all_entities, kg_embedding_model=kg_embedding_model,
#                                                  triples=triples, corrupt_suject=False, k=k)
#
#     all_hits = hits_at_k_subject_based + hits_at_k_object_based
#     hits_at_k = all_hits / (2. * triples.size)
#     stop = timeit.default_timer()
#     log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))
#
#     return hits_at_k
#
#
# def _compute_metrics(all_entities, kg_embedding_model, triples, corrupt_suject, device, k=10):
#     start = timeit.default_timer()
#     ranks = []
#     count_in_top_k = 0.
#
#     kg_embedding_model = kg_embedding_model.to(device)
#
#     column_to_maintain_offsets, corrupted_column_offsets, concatenate_fct = get_stratey_for_corrupting(
#         corrupt_suject=corrupt_suject)
#
#     start_of_columns_to_maintain, end_of_columns_to_maintain = column_to_maintain_offsets
#     start_of_corrupted_column, end_of_corrupted_column = corrupted_column_offsets
#
#     # Corrupt triples
#     for row_nmbr, row in enumerate(triples):
#
#         candidate_entities = np.delete(arr=all_entities,
#                                        obj=row[end_of_corrupted_column - 1:end_of_corrupted_column])
#
#         # Extract current test tuple: Either (subject,predicate) or (predicate,object)
#         tuple = np.reshape(a=triples[row_nmbr, start_of_columns_to_maintain:end_of_columns_to_maintain],
#                            newshape=(1, 2))
#         # Copy current test tuple
#         tuples = np.repeat(a=tuple, repeats=candidate_entities.shape[0], axis=0)
#
#         corrupted = concatenate_fct(candidate_entities=candidate_entities, tuples=tuples)
#         corrupted = torch.tensor(corrupted, dtype=torch.long, device=device)
#         scores_of_corrupted = kg_embedding_model.predict(corrupted)
#         pos_triple = np.array(row)
#         pos_triple = np.expand_dims(a=pos_triple, axis=0)
#         pos_triple = torch.tensor(pos_triple, dtype=torch.long, device=device)
#
#         score_of_positive = kg_embedding_model.predict(pos_triple)
#
#         scores = np.append(arr=scores_of_corrupted, values=score_of_positive)
#         indice_of_pos = scores.size - 1
#
#         sorted_score_indices = np.argsort(a=scores)
#
#         # Get index of first occurence that fulfills the condition
#         rank_of_positive = np.where(sorted_score_indices == indice_of_pos)[0][0]
#         ranks.append(rank_of_positive)
#
#         if rank_of_positive < k:
#             count_in_top_k += 1.
#
#     return ranks, count_in_top_k


def _compute_hits_at_k_new(hits_at_k_dict, rank_of_positive_subject_based, rank_of_positive_object_based):
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


def compute_metrics(all_entities, kg_embedding_model, triples, device):
    start = timeit.default_timer()
    ranks = []
    hits_at_k_dict = {k: [] for k in [1, 3, 5, 10]}

    kg_embedding_model = kg_embedding_model.to(device)

    # Corrupt triples
    for row_nmbr, row in enumerate(triples):
        candidate_entities_subject_based = np.delete(arr=all_entities, obj=row[0:1])
        candidate_entities_subject_based = np.reshape(candidate_entities_subject_based, newshape=(-1, 1))
        candidate_entities_object_based = np.delete(arr=all_entities, obj=row[2:3])
        candidate_entities_object_based = np.reshape(candidate_entities_object_based, newshape=(-1, 1))

        # Extract current test tuple: Either (subject,predicate) or (predicate,object)
        tuple_subject_based = np.reshape(a=triples[row_nmbr, 1:3], newshape=(1, 2))
        tuple_object_based = np.reshape(a=triples[row_nmbr, 0:2], newshape=(1, 2))

        # Copy current test tuple
        tuples_subject_based = np.repeat(a=tuple_subject_based, repeats=candidate_entities_subject_based.shape[0],
                                         axis=0)
        tuples_object_based = np.repeat(a=tuple_object_based, repeats=candidate_entities_object_based.shape[0], axis=0)

        corrupted_subject_based = np.concatenate([candidate_entities_subject_based, tuples_subject_based], axis=1)
        corrupted_subject_based = torch.tensor(corrupted_subject_based, dtype=torch.long, device=device)

        corrupted_object_based = np.concatenate([tuples_object_based, candidate_entities_object_based], axis=1)
        corrupted_object_based = torch.tensor(corrupted_object_based, dtype=torch.long, device=device)

        scores_of_corrupted_subjects = kg_embedding_model.predict(corrupted_subject_based)
        scores_of_corrupted_objects = kg_embedding_model.predict(corrupted_object_based)

        pos_triple = np.array(row)
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
        rank_of_positive_subject_based = np.where(sorted_score_indices_subject_based == indice_of_pos_subject_based)[0][
            0]
        ranks.append(rank_of_positive_subject_based)

        rank_of_positive_object_based = np.where(sorted_score_indices_object_based == indice_of_pos_object_based)[0][0]
        ranks.append(rank_of_positive_object_based)

        # Compute hits@k for k in {1,3,5,10}
        hits_at_k_dict.update(
            _compute_hits_at_k_new(hits_at_k_dict, rank_of_positive_subject_based=rank_of_positive_subject_based,
                                   rank_of_positive_object_based=rank_of_positive_object_based))

    mean_rank = np.mean(ranks)

    for k, value in hits_at_k_dict.items():
        hits_at_k_dict[k] = np.mean(value)

    stop = timeit.default_timer()
    log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))

    return mean_rank, hits_at_k_dict
