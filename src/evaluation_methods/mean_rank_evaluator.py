# -*- coding: utf-8 -*-
import numpy as np
import logging
import timeit
from evaluation_methods.abstract_evaluator import AbstractEvaluator
from utilities.constants import MEAN_RANK
from utilities.evaluation_utils.evaluation_helper import get_stratey_for_corrupting

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MeanRankEvaluator(AbstractEvaluator):
    METRIC = MEAN_RANK

    def _compute_ranks(self, kg_embedding_model, triples, corrupt_suject):
        start = timeit.default_timer()
        num_triples = len(triples)
        ranks = []

        column_to_maintain_offsets, corrupted_column_offsets, concatenate_fct = get_stratey_for_corrupting(
            corrupt_suject=corrupt_suject)

        start_of_columns_to_maintain, end_of_columns_to_maintain = column_to_maintain_offsets
        start_of_corrupted_colum, end_of_corrupted_column = corrupted_column_offsets

        # Corrupt triples
        for row in range(len(triples)):
            tuple = np.reshape(a=triples[row, start_of_columns_to_maintain:end_of_columns_to_maintain], newshape=(1, 2))
            tuples = np.repeat(a=tuple, repeats=num_triples - 1, axis=0)
            candidate_entities = np.delete(arr=triples, obj=row, axis=0)[:,
                                 start_of_corrupted_colum:end_of_corrupted_column]

            corrupted = concatenate_fct(candidate_entities=candidate_entities, tuples=tuples)
            scores_of_corrupted = kg_embedding_model.predict(corrupted)
            pos_triple = np.array(triples[row])
            pos_triple= np.expand_dims(a=pos_triple,axis=0)

            score_of_original = kg_embedding_model.predict(pos_triple)
            scores = np.append(arr=scores_of_corrupted, values=score_of_original)
            scores = np.sort(a=scores)
            # Get index of first occurence that fulfills the condition
            ranks.append(np.where(scores == score_of_original)[0][0])

        stop = timeit.default_timer()
        log.info("Evaluation took %s seconds \n" % (str(round(stop - start))))

        return ranks

    def start_evaluation(self, test_data, kg_embedding_model):
        """

        :param test_data:
        :param kg_embedding_model:
        :return:
        """

        ranks = self._compute_ranks(kg_embedding_model=kg_embedding_model, triples=test_data, corrupt_suject=True)
        ranks += self._compute_ranks(kg_embedding_model=kg_embedding_model, triples=test_data, corrupt_suject=False)
        mean_rank = np.mean(ranks)

        return mean_rank, MEAN_RANK
