# -*- coding: utf-8 -*-

"""Unittest for training utilities."""

import unittest
from typing import Type

import numpy as np
import torch

from pykeen.models import Model, TransE
from pykeen.training.utils import lazy_compile_random_batches


class LossTensorTest(unittest.TestCase):
    """Test label smoothing."""

    model_cls: Type[Model] = TransE
    embedding_dim: int = 8

    def setUp(self):
        """Set up the loss tensor tests."""
        self.triples = np.array(
            [
                ['peter', 'likes', 'chocolate_cake'],
                ['chocolate_cake', 'isA', 'dish'],
                ['susan', 'likes', 'pizza'],
                ['peter', 'likes', 'susan'],
            ],
            dtype=str,
        )

        self.labels = torch.tensor([
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 1.],
        ])

        self.predictions = torch.tensor([
            [1., 0., 1., 1., 1.],
            [1., 1., 1., 0., 1.],
            [0., 1., 1., 1., 0.],
        ])


class BatchCompilationTest(unittest.TestCase):
    """Test compilation of random batches."""

    batch_size: int = 64
    num_samples: int = 256 + batch_size // 2  # to check whether the method works for incomplete batches
    num_entities: int = 10

    def setUp(self) -> None:
        """Set up the test case with a fixed random seed."""
        self.random = np.random.RandomState(seed=42)

    def test_lazy_compile_random_batches(self):
        """Test method lazy_compile_random_batches."""
        indices = np.arange(self.num_samples)
        input_array = self.random.randint(low=0, high=self.num_entities, size=(self.num_samples, 2), dtype=np.int64)
        targets = []
        for _ in range(self.num_samples):
            targets.append(list(set(self.random.randint(low=0, high=self.num_entities, size=(5,), dtype=np.int64))))
        target_array = np.asarray(targets)

        def _batch_compiler(batch_indices):
            return input_array[batch_indices], target_array[batch_indices]

        iterator = lazy_compile_random_batches(
            indices=indices,
            batch_size=self.batch_size,
            batch_compiler=_batch_compiler,
        )
        all_elements = list(iterator)
        for input_batch, target_batch in all_elements[:-1]:
            self.assertEqual(input_batch.shape, (self.batch_size, 2))
            self.assertEqual(target_batch.shape, (self.batch_size,))
        last_input_batch, last_target_batch = all_elements[-1]
        self.assertEqual(last_input_batch.shape, (self.num_samples % self.batch_size, 2))
        self.assertEqual(last_target_batch.shape, (self.num_samples % self.batch_size,))
