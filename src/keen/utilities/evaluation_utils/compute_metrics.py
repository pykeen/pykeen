# -*- coding: utf-8 -*-
import logging
import timeit

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
