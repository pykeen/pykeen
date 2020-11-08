# -*- coding: utf-8 -*-

"""Unittest for training utilities."""

import unittest
from typing import Type

import numpy as np
import torch

from pykeen.losses import MarginRankingLoss
from pykeen.models import TransE
from pykeen.models.base import Model
from pykeen.training.lcwa import LCWATrainingLoop
from pykeen.training.utils import apply_label_smoothing, lazy_compile_random_batches
from pykeen.triples import TriplesFactory


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
            dtype=np.str,
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

    def test_lcwa_margin_ranking_loss_helper(self):
        """Test if output is correct for the LCWA training loop use case."""
        factory = TriplesFactory(triples=self.triples)

        loss_cls = MarginRankingLoss(
            margin=0,
            reduction='sum',
        )

        model = TransE(
            factory,
            embedding_dim=8,
            preferred_device='cpu',
            loss=loss_cls,
        )

        loop = LCWATrainingLoop(model=model)
        loss = loop._mr_loss_helper(predictions=self.predictions, labels=self.labels)
        self.assertEqual(14, loss)

        loss_cls = MarginRankingLoss(
            margin=0,
            reduction='mean',
        )

        model = TransE(
            factory,
            embedding_dim=8,
            preferred_device='cpu',
            loss=loss_cls,
        )

        loop = LCWATrainingLoop(model=model)
        loss = loop._mr_loss_helper(predictions=self.predictions, labels=self.labels)
        self.assertEqual(1, loss)


class LabelSmoothingTest(unittest.TestCase):
    """Test label smoothing."""

    batch_size: int = 16
    num_entities: int = 32
    epsilon: float = 0.1
    relative_tolerance: float = 1.e-4  # larger tolerance for float32

    def setUp(self) -> None:
        """Set up the test case with a fixed random seed."""
        self.random = np.random.RandomState(seed=42)

    def test_lcwa_label_smoothing(self):
        """Test if output is correct for the LCWA training loop use case."""
        # Create dummy dense labels
        labels = torch.zeros(self.batch_size, self.num_entities)
        for i in range(self.batch_size):
            labels[i, self.random.randint(self.num_entities)] = 1.0
        # Check if labels form a probability distribution
        np.testing.assert_allclose(torch.sum(labels, dim=1).numpy(), 1.0)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels=labels, epsilon=self.epsilon, num_classes=self.num_entities)
        # Check if smooth labels form probability distribution
        np.testing.assert_allclose(torch.sum(smooth_labels, dim=1).numpy(), 1.0, rtol=self.relative_tolerance)

    def test_slcwa_label_smoothing(self):
        """Test if output is correct for the sLCWA training loop use case."""
        # Create dummy sLCWA labels
        ones = torch.ones(self.batch_size, 1)
        zeros = torch.zeros(self.batch_size, 1)
        labels = torch.cat([ones, zeros], dim=0)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels=labels, epsilon=self.epsilon, num_classes=self.num_entities)
        exp_true = 1.0 - self.epsilon
        np.testing.assert_allclose(smooth_labels[:self.batch_size], exp_true, rtol=self.relative_tolerance)
        exp_false = self.epsilon / (self.num_entities - 1.)
        np.testing.assert_allclose(smooth_labels[self.batch_size:], exp_false, rtol=self.relative_tolerance)


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
        input_array = self.random.randint(low=0, high=self.num_entities, size=(self.num_samples, 2), dtype=np.long)
        targets = []
        for _ in range(self.num_samples):
            targets.append(list(set(self.random.randint(low=0, high=self.num_entities, size=(5,), dtype=np.long))))
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
