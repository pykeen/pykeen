# -*- coding: utf-8 -*-

"""Unittest for training utilities."""

import unittest
from typing import Type

import numpy as np
import torch

from pykeen.models import Model, TransE


class LossTensorTest(unittest.TestCase):
    """Test label smoothing."""

    model_cls: Type[Model] = TransE
    embedding_dim: int = 8

    def setUp(self):
        """Set up the loss tensor tests."""
        self.triples = np.array(
            [
                ["peter", "likes", "chocolate_cake"],
                ["chocolate_cake", "isA", "dish"],
                ["susan", "likes", "pizza"],
                ["peter", "likes", "susan"],
            ],
            dtype=str,
        )

        self.labels = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.predictions = torch.tensor(
            [
                [1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
