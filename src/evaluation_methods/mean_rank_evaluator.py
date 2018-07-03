# -*- coding: utf-8 -*-
import numpy as np

from evaluation_methods.abstract_evaluator import AbstractEvaluator
from utilities.constants import MEAN_RANK
from utilities.eval_utilities import get_stratey_for_corrupting


class MeanRankEvaluator(AbstractEvaluator):
    METRIC = MEAN_RANK

    def _compute_ranks(self, kg_embedding_model, data, corrupt_suject):
        num_triples = len(data)
        ranks = []

        column_to_maintain_offsets, corrupted_column_offsets, concatenate_fct = get_stratey_for_corrupting(
            corrupt_suject=corrupt_suject)

        start_of_columns_to_maintain, end_of_columns_to_maintain = column_to_maintain_offsets
        start_of_corrupted_colum, end_of_corrupted_column = corrupted_column_offsets

        # Corrupt triples
        for row in range(len(data)):
            tuple = np.reshape(a=data[row, start_of_columns_to_maintain:end_of_columns_to_maintain], newshape=(1, 2))
            tuples = np.repeat(a=tuple, repeats=num_triples - 1, axis=0)
            candidate_entities = np.delete(arr=data, obj=row, axis=0)[:,
                                 start_of_corrupted_colum:end_of_corrupted_column]

            corrupted = concatenate_fct(candidate_entities=candidate_entities, tuples=tuples)

            scores_of_corrupted = np.apply_along_axis(kg_embedding_model.predict, axis=1, arr=corrupted)
            score_of_original = kg_embedding_model.predict(data[row])
            scores = np.append(arr=scores_of_corrupted, values=score_of_original)
            scores = np.sort(a=scores)
            # Get index of first occurence that fulfills the condition
            ranks.append(np.where(scores == score_of_original)[0][0])

        return ranks

    def start_evaluation(self, test_data, kg_embedding_model):
        """

        :param test_data:
        :param kg_embedding_model:
        :return:
        """

        ranks = self._compute_ranks(kg_embedding_model=kg_embedding_model, data=test_data, corrupt_suject=True)
        ranks += self._compute_ranks(kg_embedding_model=kg_embedding_model, data=test_data, corrupt_suject=False)
        mean_rank = np.mean(ranks)

        return mean_rank, MEAN_RANK
