# -*- coding: utf-8 -*-

"""Tests for uncertainty workflows."""

from typing import Callable, Tuple

import torch

from pykeen.models import ERMLPE, TransE
from pykeen.models.uncertainty import (
    UncertainPrediction,
    predict_h_uncertain,
    predict_hrt_uncertain,
    predict_r_uncertain,
    predict_t_uncertain,
)
from tests import cases


class UncertaintyFailureTest(cases.PredictBaseTestCase):
    """Test for when uncertainty can't be assessed with MC method."""

    model_cls = TransE
    model_kwargs = {}

    def test_missing_dropout(self):
        """Test that a value error is run if the model has no dropout."""
        with self.assertRaises(ValueError):
            predict_hrt_uncertain(model=self.model, hrt_batch=self.batch)


class UncertaintyPredictionTestCase(cases.PredictBaseTestCase):
    """Tests for uncertainty prediction."""

    model_cls = ERMLPE  # this model does indeed have dropouts!
    model_kwargs = {
        "embedding_dim": 2,
        "hidden_dim": 3,
    }
    num_samples: int = 3

    def _test_predict_uncertain(
        self,
        method: Callable[..., Tuple[torch.FloatTensor, torch.FloatTensor]],
        expected_shape: Tuple[int, ...],
        **kwargs,
    ):
        """General testing of uncertainty prediction."""
        result = method(
            model=self.model,
            num_samples=self.num_samples,
            **kwargs,
        )
        assert isinstance(result, UncertainPrediction)
        assert (result.uncertainty >= 0).all()
        assert result.score.shape == expected_shape
        assert result.uncertainty.shape == expected_shape

    def test_predict_hrt_uncertain(self):
        """Test predict_hrt_uncertain."""
        self._test_predict_uncertain(
            method=predict_hrt_uncertain,
            expected_shape=(self.batch_size, 1),
            hrt_batch=self.batch,
        )

    def test_predict_h_uncertain(self):
        """Test predict_h_uncertain."""
        self._test_predict_uncertain(
            method=predict_h_uncertain,
            expected_shape=(self.batch_size, self.factory.num_entities),
            rt_batch=self.batch[:, 1:],
        )

    def test_predict_r_uncertain(self):
        """Test predict_r_uncertain."""
        self._test_predict_uncertain(
            method=predict_r_uncertain,
            expected_shape=(self.batch_size, self.factory.num_relations),
            ht_batch=self.batch[:, [0, 2]],
        )

    def test_predict_t_uncertain(self):
        """Test predict_t_uncertain."""
        self._test_predict_uncertain(
            method=predict_t_uncertain,
            expected_shape=(self.batch_size, self.factory.num_entities),
            hr_batch=self.batch[:, :2],
        )
