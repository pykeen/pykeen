# -*- coding: utf-8 -*-

"""Test RankedBasedEvaluator."""

import unittest
import numpy as np
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.preprocessing.triples_preprocessing_utils.basic_triple_utils import create_entity_and_relation_mappings


class RankedBasedEvaluatorTests(unittest.TestCase):
    def test_compute_rank(self):
        scores_of_corrupted_subjects = np.array([2, 3, 5, 10], dtype=np.float)
        scores_of_corrupted_objects = np.array([1, 1, 3, 4], dtype=np.float)

        score_of_positive = np.array([3], dtype=np.float)

        rank_of_positive_subject_based = scores_of_corrupted_subjects.shape[0] - \
                                         np.greater_equal(scores_of_corrupted_subjects, score_of_positive).sum()

        rank_of_positive_object_based = scores_of_corrupted_objects.shape[0] - \
                                        np.greater_equal(scores_of_corrupted_objects, score_of_positive).sum()

        self.assertEqual(rank_of_positive_subject_based, 1)
        self.assertEqual(rank_of_positive_object_based, 2)
