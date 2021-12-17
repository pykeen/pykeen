"""Tests for prediction workflows."""
from typing import Callable, Tuple

import torch

import pykeen.models.predict
import tests.cases


class UncertaintyPredictionTestCase(tests.cases.PredictBaseTestCase):
    """Tests for uncertainty prediction."""

    num_samples: int = 3

    def _test_predict_uncertain(
        self,
        method: Callable[..., Tuple[torch.FloatTensor, torch.FloatTensor]],
        expected_shape: Tuple[int, ...],
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """General testing of uncertainty prediction."""
        result = method(
            model=self.model,
            num_samples=self.num_samples,
            **kwargs,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        mean, std = result
        assert (std >= 0).all()
        assert mean.shape == expected_shape
        assert std.shape == expected_shape

    def test_predict_hrt_uncertain(self):
        """Test predict_hrt_uncertain."""
        self._test_predict_uncertain(
            method=pykeen.models.predict.predict_hrt_uncertain,
            expected_shape=(self.batch_size, 1),
            hrt_batch=self.batch,
        )

    def test_predict_t_uncertain(self):
        """Test predict_t_uncertain."""
        self._test_predict_uncertain(
            method=pykeen.models.predict.predict_t_uncertain,
            expected_shape=(self.batch_size, self.factory.num_entities),
            hrt_batch=self.batch[:, :2],
        )
