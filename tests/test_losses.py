# -*- coding: utf-8 -*-

"""Test the PyKEEN custom loss functions."""

import unittest
from typing import Any, Mapping, Optional, Type

import torch
from torch.nn import functional

from pykeen.losses import BCEAfterSigmoidLoss, CrossEntropyLoss, Loss, NSSALoss, SoftplusLoss
from pykeen.pipeline import PipelineResult, pipeline


class _LossTests:
    """Base unittest for loss functions."""

    #: The class
    cls: Type[Loss]

    #: Constructor keyword arguments
    kwargs: Optional[Mapping[str, Any]] = None

    #: The loss instance
    instance: Loss

    #: The batch size
    batch_size: int = 3

    def setUp(self) -> None:
        """Initialize the instance."""
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        self.instance = self.cls(**kwargs)

    def _check_loss_value(self, loss_value: torch.FloatTensor) -> None:
        """Check loss value dimensionality, and ability for backward."""
        # test reduction
        assert loss_value.ndim == 0

        # Test backward
        loss_value.backward()


class _LabelLossTests(_LossTests):
    """Base unit test for label-based losses."""

    #: The number of entities.
    num_entities: int = 17

    def test_label_loss(self):
        """Test ``forward(logits, labels)``."""
        logits = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        labels = functional.normalize(torch.rand(self.batch_size, self.num_entities, requires_grad=False), p=1, dim=-1)
        loss_value = self.instance.forward(
            logits=logits,
            labels=labels,
        )
        self._check_loss_value(loss_value)


class _PairLossTests(_LossTests):
    """Base unit test for pair-wise losses."""

    #: The number of negative samples
    num_negatives: int = 5

    def test_pair_loss(self):
        """Test ``forward(pos_scores, neg_scores)``."""
        pos_scores = torch.rand(self.batch_size, 1, requires_grad=True)
        neg_scores = torch.rand(self.batch_size, self.num_negatives, requires_grad=True)
        loss_value = self.instance.forward(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
        )
        self._check_loss_value(loss_value)


class CrossEntropyLossTests(_LabelLossTests, unittest.TestCase):
    """Unit test for CrossEntropyLoss."""

    cls = CrossEntropyLoss


class BCEAfterSigmoidLossTests(_LabelLossTests, unittest.TestCase):
    """Unit test for BCEAfterSigmoidLoss."""

    cls = BCEAfterSigmoidLoss


class SoftplusLossTests(_LabelLossTests, unittest.TestCase):
    """Unit test for SoftplusLoss."""

    cls = SoftplusLoss


class NSSALossTests(_PairLossTests, unittest.TestCase):
    """Unit test for NSSALoss."""

    cls = NSSALoss
    kwargs = {
        'margin': 1.,
        'adversarial_temperature': 1.,
    }


class TestCustomLossFunctions(unittest.TestCase):
    """Test the custom loss functions."""

    def test_negative_sampling_self_adversarial_loss(self):
        """Test the negative sampling self adversarial loss function."""
        loss_fct = NSSALoss(margin=1., adversarial_temperature=1.)

        pos_scores = torch.tensor([0., 0., -0.5, -0.5])
        neg_scores = torch.tensor([0., 0., -1., -1.])

        # ≈ result of softmax
        weights = torch.tensor([0.37, 0.37, 0.13, 0.13])

        # neg_distances - margin = [-1., -1., 0., 0.]
        # sigmoids ≈ [0.27, 0.27, 0.5, 0.5]
        log_sigmoids = torch.tensor([-1.31, -1.31, -0.69, -0.69])
        intermediate = weights * log_sigmoids
        neg_loss = torch.mean(intermediate, dim=-1)

        # pos_distances = [0., 0., 0.5, 0.5]
        # margin - pos_distances = [1. 1., 0.5, 0.5]
        # ≈ result of sigmoid
        # sigmoids ≈ [0.73, 0.73, 0.62, 0.62]
        log_sigmoids = torch.tensor([-0.31, -0.31, -0.48, -0.48])
        pos_loss = torch.mean(log_sigmoids)

        # expected_loss ≈ 0.34
        expected_loss = (-pos_loss - neg_loss) / 2.

        loss = loss_fct(pos_scores, neg_scores).item()

        self.assertAlmostEqual(expected_loss, 0.34, delta=0.02)
        self.assertAlmostEqual(expected_loss, loss, delta=0.02)

    def test_pipeline(self):
        """Test the pipeline on RotatE with negative sampling self adversarial loss and nations."""
        loss = NSSALoss
        loss_kwargs = {"margin": 1., "adversarial_temperature": 1.}
        pipeline_results = pipeline(
            model='RotatE',
            dataset='nations',
            loss=loss,
            loss_kwargs=loss_kwargs,
        )
        self.assertIsInstance(pipeline_results, PipelineResult)
        self.assertIsInstance(pipeline_results.model.loss, loss)
        self.assertEqual(pipeline_results.model.loss.margin, 1.)
        self.assertEqual(pipeline_results.model.loss.adversarial_temperature, 1.)
