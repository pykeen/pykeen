# coding=utf-8

"""Unittest for training utilities."""

import unittest

import numpy
import torch

from poem.training.utils import apply_label_smoothing, lazy_compile_random_batches


class LabelSmoothingTest(unittest.TestCase):
    """Test label smoothing."""

    batch_size = 16
    num_entities = 32
    epsilon = 0.1
    relative_tolerance = 1.e-4  # larger tolerance for float32

    def test_cwa_label_smoothing(self):
        """Test if output is correct for the CWA training loop use case."""
        # Create dummy dense labels
        labels = torch.zeros(self.batch_size, self.num_entities)
        for i in range(self.batch_size):
            labels[i, numpy.random.randint(self.num_entities)] = 1.0
        # Check if labels form a probability distribution
        numpy.testing.assert_allclose(torch.sum(labels, dim=1).numpy(), 1.0)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels=labels, epsilon=self.epsilon, num_classes=self.num_entities)
        # Check if smooth labels form probability distribution
        numpy.testing.assert_allclose(torch.sum(smooth_labels, dim=1).numpy(), 1.0, rtol=self.relative_tolerance)

    def test_owa_label_smoothing(self):
        """Test if output is correct for the OWA training loop use case."""
        # Create dummy OWA labels
        ones = torch.ones(self.batch_size, 1)
        zeros = torch.zeros(self.batch_size, 1)
        labels = torch.cat([ones, zeros], dim=0)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels=labels, epsilon=self.epsilon, num_classes=self.num_entities)
        exp_true = 1.0 - self.epsilon
        numpy.testing.assert_allclose(smooth_labels[:self.batch_size], exp_true, rtol=self.relative_tolerance)
        exp_false = self.epsilon / (self.num_entities - 1.)
        numpy.testing.assert_allclose(smooth_labels[self.batch_size:], exp_false, rtol=self.relative_tolerance)


class BatchCompilationTest(unittest.TestCase):
    """Test compilation of random batches."""

    batch_size = 64
    num_samples = 256 + batch_size // 2  # to check whether the method works for incomplete batches
    num_entities = 10

    def test_lazy_compile_random_batches(self):
        """Test method lazy_compile_random_batches."""
        indices = numpy.arange(self.num_samples)
        input_array = numpy.random.randint(low=0, high=self.num_entities, size=(self.num_samples, 2), dtype=numpy.long)
        targets = []
        for i in range(self.num_samples):
            targets.append(list(set(numpy.random.randint(low=0, high=self.num_entities, size=(5,), dtype=numpy.long))))
        target_array = numpy.asarray(targets)
        iterator = lazy_compile_random_batches(
            indices=indices,
            input_array=input_array,
            target_array=target_array,
            batch_size=self.batch_size
        )
        all_elements = list(iterator)
        for input_batch, target_batch in all_elements[:-1]:
            self.assertEqual(input_batch.shape, (self.batch_size, 2))
            self.assertEqual(target_batch.shape, (self.batch_size,))
        last_input_batch, last_target_batch = all_elements[-1]
        self.assertEqual(last_input_batch.shape, (self.num_samples % self.batch_size, 2))
        self.assertEqual(last_target_batch.shape, (self.num_samples % self.batch_size,))
