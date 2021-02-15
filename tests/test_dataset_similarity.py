# -*- coding: utf-8 -*-

"""Tests for remixing and other triples reorganization."""

import unittest

import torch

from pykeen.triples import CoreTriplesFactory
from pykeen.triples.triples_factory import splits_similarity, splits_steps


class TestRemix(unittest.TestCase):
    """Tests for remix functions."""

    def test_splits_similarity(self):
        """Test the similarity calculation."""
        a_train = torch.as_tensor([
            [1, 1, 2],
            [2, 1, 3],
            [1, 2, 3],
            [4, 1, 5],
            [5, 1, 6],
        ])
        a_test = torch.as_tensor([
            [4, 2, 6],
        ])
        b_train = torch.as_tensor([
            [1, 1, 2],
            [2, 1, 3],
            [1, 2, 3],
            [4, 1, 5],
            [4, 2, 6],
        ])
        b_test = torch.as_tensor([
            [5, 1, 6],
        ])

        a_train_tf = CoreTriplesFactory.create(a_train)
        a_test_tf = CoreTriplesFactory.create(a_test)
        b_train_tf = CoreTriplesFactory.create(b_train)
        b_test_tf = CoreTriplesFactory.create(b_test)

        steps = splits_steps([a_train_tf, a_test_tf], [b_train_tf, b_test_tf])
        self.assertEqual(2, steps)

        similarity = splits_similarity([a_train_tf, a_test_tf], [b_train_tf, b_test_tf])
        self.assertEqual(1 - steps / 6, similarity)
