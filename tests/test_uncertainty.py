"""Tests for uncertainty workflows."""

from collections.abc import Callable

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.models import ERMLPE, TransE
from pykeen.models.uncertainty import (
    MissingDropoutError,
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
        with pytest.raises(MissingDropoutError):
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
        method: Callable[..., tuple[torch.FloatTensor, torch.FloatTensor]],
        expected_shape: tuple[int, ...],
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


@pytest.mark.parametrize(
    ("uncertain_method", "deterministic_method_name", "batch_name", "columns"),
    [
        (predict_hrt_uncertain, "predict_hrt", "hrt_batch", [0, 1, 2]),
        (predict_h_uncertain, "predict_h", "rt_batch", [1, 2]),
        (predict_t_uncertain, "predict_t", "hr_batch", [0, 1]),
    ],
)
def test_uncertain_prediction_with_inverse_triples(
    uncertain_method: Callable[..., UncertainPrediction],
    deterministic_method_name: str,
    batch_name: str,
    columns: list[int],
):
    """Test that uncertainty prediction maps relation IDs like its deterministic counterpart."""
    dataset = Nations(create_inverse_triples=True)
    # disable dropout, so that the MC samples coincide with the deterministic prediction
    model = ERMLPE(
        triples_factory=dataset.training,
        embedding_dim=2,
        hidden_dim=3,
        random_seed=0,
        input_dropout=0.0,
        hidden_dropout=0.0,
    )
    batch = dataset.testing.mapped_triples[:3, columns]
    copy = batch.clone()

    result = uncertain_method(model=model, num_samples=3, **{batch_name: batch})

    # the input batch must not be modified
    assert torch.equal(batch, copy)
    # ... and the same relations have to be scored as by the deterministic method
    expected = getattr(model, deterministic_method_name)(batch.clone())
    assert torch.allclose(result.score, expected, atol=1.0e-06)
