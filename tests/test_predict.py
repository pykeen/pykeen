# -*- coding: utf-8 -*-

"""Tests for prediction workflows."""

from typing import Callable, Tuple

import torch

import pykeen.models.predict
from tests import cases


class UncertaintyPredictionTestCase(cases.PredictBaseTestCase):
    """Tests for uncertainty prediction."""

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
        assert isinstance(result, pykeen.models.predict.UncertainPrediction)
        assert (result.uncertainty >= 0).all()
        assert result.score.shape == expected_shape
        assert result.uncertainty.shape == expected_shape

    def test_predict_hrt_uncertain(self):
        """Test predict_hrt_uncertain."""
        self._test_predict_uncertain(
            method=pykeen.models.predict.predict_hrt_uncertain,
            expected_shape=(self.batch_size, 1),
            hrt_batch=self.batch,
        )

    def test_predict_h_uncertain(self):
        """Test predict_h_uncertain."""
        self._test_predict_uncertain(
            method=pykeen.models.predict.predict_h_uncertain,
            expected_shape=(self.batch_size, self.factory.num_entities),
            rt_batch=self.batch[:, 1:],
        )

    def test_predict_r_uncertain(self):
        """Test predict_r_uncertain."""
        self._test_predict_uncertain(
            method=pykeen.models.predict.predict_r_uncertain,
            expected_shape=(self.batch_size, self.factory.num_relations),
            ht_batch=self.batch[:, [0, 2]],
        )

    def test_predict_t_uncertain(self):
        """Test predict_t_uncertain."""
        self._test_predict_uncertain(
            method=pykeen.models.predict.predict_t_uncertain,
            expected_shape=(self.batch_size, self.factory.num_entities),
            hr_batch=self.batch[:, :2],
        )
