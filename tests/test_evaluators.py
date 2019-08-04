# -*- coding: utf-8 -*-

"""Test the evaluators."""

import unittest

import numpy as np


class RankedBasedEvaluatorTests(unittest.TestCase):
    """Test the rank-based evaluator."""

    def test_compute_rank(self):
        """Test the compute_rank() function."""
        scores_of_corrupted_subjects = np.array([2, 3, 5, 10], dtype=np.float)
        scores_of_corrupted_objects = np.array([1, 1, 3, 4], dtype=np.float)

        score_of_positive = np.array([3], dtype=np.float)

        sub_shape = scores_of_corrupted_subjects.shape[0]
        sub_scores = np.greater_equal(scores_of_corrupted_subjects, score_of_positive).sum()
        rank_of_positive_subject_based = sub_shape - sub_scores

        obj_shape = scores_of_corrupted_objects.shape[0]
        obj_scores = np.greater_equal(scores_of_corrupted_objects, score_of_positive).sum()
        rank_of_positive_object_based = obj_shape - obj_scores

        self.assertEqual(rank_of_positive_subject_based, 1)
        self.assertEqual(rank_of_positive_object_based, 2)
