# -*- coding: utf-8 -*-

"""Test the PyKEEN custom loss functions."""

import unittest

import torch

from pykeen.losses import BCEAfterSigmoidLoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NSSALoss, SoftplusLoss
from pykeen.pipeline import PipelineResult, pipeline
from tests import cases


class CrossEntropyLossTests(cases.SetwiseLossTestCase):
    """Unit test for CrossEntropyLoss."""

    cls = CrossEntropyLoss


class BCEAfterSigmoidLossTests(cases.PointwiseLossTestCase):
    """Unit test for BCEAfterSigmoidLoss."""

    cls = BCEAfterSigmoidLoss


class SoftplusLossTests(cases.PointwiseLossTestCase):
    """Unit test for SoftplusLoss."""

    cls = SoftplusLoss


class NSSALossTests(cases.SetwiseLossTestCase):
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
        self.assertIs(loss_fct._reduction_method, torch.mean)

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
            training_kwargs=dict(use_tqdm=False),
        )
        self.assertIsInstance(pipeline_results, PipelineResult)
        self.assertIsInstance(pipeline_results.model.loss, loss)
        self.assertEqual(pipeline_results.model.loss.margin, 1.)
        self.assertEqual(pipeline_results.model.loss.adversarial_temperature, 1.)


class BCEWithLogitsLossTestCase(cases.PointwiseLossTestCase):
    """Tests for binary cross entropy (stable) loss."""

    cls = BCEWithLogitsLoss


class MSELossTestCase(cases.PointwiseLossTestCase):
    """Tests for mean square error loss."""

    cls = MSELoss

# TODO reimplement then test MarginRankingLoss using PR #18 solution
