# -*- coding: utf-8 -*-

"""Test the PyKEEN custom loss functions."""

import unittest

import numpy as np
import torch
import unittest_templates

import pykeen.losses
from pykeen.losses import Loss, NSSALoss, PairwiseLoss, PointwiseLoss, SetwiseLoss, apply_label_smoothing
from pykeen.pipeline import PipelineResult, pipeline
from tests import cases


class CrossEntropyLossTests(cases.SetwiseLossTestCase):
    """Unit test for CrossEntropyLoss."""

    cls = pykeen.losses.CrossEntropyLoss


class BCEAfterSigmoidLossTests(cases.PointwiseLossTestCase):
    """Unit test for BCEAfterSigmoidLoss."""

    cls = pykeen.losses.BCEAfterSigmoidLoss


class DoubleMarginLossTests(cases.PointwiseLossTestCase):
    """Unit test for DoubleMarginLoss."""

    cls = pykeen.losses.DoubleMarginLoss


class FocalLossTests(cases.PointwiseLossTestCase):
    """Tests for focal loss."""

    cls = pykeen.losses.FocalLoss


class SoftplusLossTests(cases.PointwiseLossTestCase):
    """Unit test for SoftplusLoss."""

    cls = pykeen.losses.SoftplusLoss


class PointwiseHingeLossTests(cases.PointwiseLossTestCase):
    """Unit test for the pointwise hinge loss."""

    cls = pykeen.losses.PointwiseHingeLoss


class SoftPointwiseHingeLossTests(cases.PointwiseLossTestCase):
    """Unit test for the soft pointwise hinge loss."""

    cls = pykeen.losses.SoftPointwiseHingeLoss


class NSSALossTests(cases.SetwiseLossTestCase):
    """Unit test for NSSALoss."""

    cls = pykeen.losses.NSSALoss
    kwargs = {
        "margin": 1.0,
        "adversarial_temperature": 1.0,
    }


class TestCustomLossFunctions(unittest.TestCase):
    """Test the custom loss functions."""

    def test_negative_sampling_self_adversarial_loss(self):
        """Test the negative sampling self adversarial loss function."""
        loss_fct = NSSALoss(margin=1.0, adversarial_temperature=1.0)
        self.assertIs(loss_fct._reduction_method, torch.mean)

        pos_scores = torch.tensor([0.0, 0.0, -0.5, -0.5])
        neg_scores = torch.tensor([0.0, 0.0, -1.0, -1.0])

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
        expected_loss = (-pos_loss - neg_loss) / 2.0

        loss = loss_fct(pos_scores, neg_scores, weights).item()

        self.assertAlmostEqual(expected_loss, 0.34, delta=0.02)
        self.assertAlmostEqual(expected_loss, loss, delta=0.02)

    def test_pipeline(self):
        """Test the pipeline on RotatE with negative sampling self adversarial loss and nations."""
        loss = NSSALoss
        loss_kwargs = {"margin": 1.0, "adversarial_temperature": 1.0}
        pipeline_results = pipeline(
            model="RotatE",
            dataset="nations",
            loss=loss,
            loss_kwargs=loss_kwargs,
            training_kwargs=dict(use_tqdm=False),
        )
        self.assertIsInstance(pipeline_results, PipelineResult)
        self.assertIsInstance(pipeline_results.model.loss, loss)
        self.assertEqual(pipeline_results.model.loss.margin, 1.0)
        self.assertEqual(pipeline_results.model.loss.inverse_softmax_temperature, 1.0)


class BCEWithLogitsLossTestCase(cases.PointwiseLossTestCase):
    """Tests for binary cross entropy (stable) loss."""

    cls = pykeen.losses.BCEWithLogitsLoss


class MSELossTestCase(cases.PointwiseLossTestCase):
    """Tests for mean square error loss."""

    cls = pykeen.losses.MSELoss


class MarginRankingLossTestCase(cases.GMRLTestCase):
    """Tests for margin ranking loss."""

    cls = pykeen.losses.MarginRankingLoss


class SoftMarginrankingLossTestCase(cases.GMRLTestCase):
    """Tests for the soft margin ranking loss."""

    cls = pykeen.losses.SoftMarginRankingLoss


class PairwiseLogisticLossTestCase(cases.GMRLTestCase):
    """Tests for the pairwise logistic loss."""

    cls = pykeen.losses.PairwiseLogisticLoss


class InfoNCELossTests(cases.SetwiseLossTestCase):
    """Unit test for InfoNCE loss."""

    cls = pykeen.losses.InfoNCELoss


class TestLosses(unittest_templates.MetaTestCase[Loss]):
    """Test that the loss functions all have tests."""

    base_cls = Loss
    base_test = cases.LossTestCase
    skip_cls = {
        PairwiseLoss,
        PointwiseLoss,
        SetwiseLoss,
        pykeen.losses.DeltaPointwiseLoss,
        pykeen.losses.MarginPairwiseLoss,
    }


class LabelSmoothingTest(unittest.TestCase):
    """Test label smoothing."""

    batch_size: int = 16
    num_entities: int = 32
    epsilon: float = 0.1
    relative_tolerance: float = 1.0e-4  # larger tolerance for float32

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
        np.testing.assert_allclose(smooth_labels[: self.batch_size], exp_true, rtol=self.relative_tolerance)
        exp_false = self.epsilon / (self.num_entities - 1.0)
        np.testing.assert_allclose(smooth_labels[self.batch_size :], exp_false, rtol=self.relative_tolerance)
