# -*- coding: utf-8 -*-

import numpy as np


def get_stratey_for_corrupting(corrupt_suject):
    if corrupt_suject:
        start_of_columns_to_maintain = 1
        end_of_columns_to_maintain = 3
        start_of_corrupted_colum = 0
        end_of_corrupted_column = 1

        concatenate_fct = _concatenate_entites_first

    else:
        start_of_columns_to_maintain = 0
        end_of_columns_to_maintain = 2
        start_of_corrupted_colum = 2
        end_of_corrupted_column = 3

        concatenate_fct = _concatenate_entites_last

    return (start_of_columns_to_maintain, end_of_columns_to_maintain), (
        start_of_corrupted_colum, end_of_corrupted_column), concatenate_fct


def _concatenate_entites_first(candidate_entities, tuples):
    candidate_entities = np.reshape(candidate_entities, newshape=(-1, 1))
    return np.concatenate([candidate_entities, tuples], axis=1)


def _concatenate_entites_last(candidate_entities, tuples):
    candidate_entities = np.reshape(candidate_entities, newshape=(-1, 1))
    return np.concatenate([tuples, candidate_entities], axis=1)
