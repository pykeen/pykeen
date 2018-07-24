# -*- coding: utf-8 -*-
import numpy as np

from evaluation_methods.abstract_evaluator import AbstractEvaluator
from utilities.constants import HITS_AT_K
from utilities.evaluation_utils.eval_utilities import get_stratey_for_corrupting


class HitsAtKEvaluator(AbstractEvaluator):
    METRIC = HITS_AT_K

    def _compute_hits_at_k(self, kg_embedding_model, data, k, corrupt_suject):
        num_triples = len(data)

        assert num_triples>=k

        column_to_maintain_offsets, corrupted_column_offsets, concatenate_fct = get_stratey_for_corrupting(
            corrupt_suject=corrupt_suject)

        start_of_columns_to_maintain, end_of_columns_to_maintain = column_to_maintain_offsets
        start_of_corrupted_colum, end_of_corrupted_column = corrupted_column_offsets

        scores_of_corrupted = []
        scores_of_originals = []

        # Corrupt triples
        for row in range(len(data)):
            tuple = np.reshape(a=data[row, start_of_columns_to_maintain:end_of_columns_to_maintain], newshape=(1, 2))
            tuples = np.repeat(a=tuple, repeats=num_triples - 1, axis=0)
            candidate_entities = np.delete(arr=data, obj=row, axis=0)[:,
                                 start_of_corrupted_colum:end_of_corrupted_column]

            corrupted = concatenate_fct(candidate_entities=candidate_entities, tuples=tuples)

            scores_of_corrupted.append(np.apply_along_axis(kg_embedding_model.predict, axis=1, arr=corrupted))
            scores_of_originals.append(kg_embedding_model.predict(data[row]))


        # Get top k
        scores_of_corrupted = np.concatenate(scores_of_corrupted,axis=-1)
        scores_of_originals = np.concatenate(scores_of_originals,axis=-1)
        all_scores = np.append(arr=scores_of_corrupted,values=scores_of_originals)
        all_scores = np.sort(a=all_scores)
        top_10_scores = all_scores[-k:]

        # TODO: Last step



        return None

