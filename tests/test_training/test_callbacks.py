"""Tests for training callbacks."""
import unittest
from typing import Any, MutableMapping
from unittest import mock

import torch

from pykeen.evaluation.evaluator import Evaluator
from pykeen.pipeline import pipeline
from pykeen.training.callbacks import EvaluationTrainingCallback

from .. import cases


class EvaluationTrainingCallbackTestCase(cases.TrainingCallbackTestCase):
    """Test for evaluation callback."""

    cls = EvaluationTrainingCallback

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["evaluation_triples"] = self.dataset.validation.mapped_triples
        return kwargs

    @unittest.skipIf(
        condition=not torch.cuda.is_available(),
        reason="automatic memory optimization only active for GPU",
    )
    def test_batch_size(self):
        """Test batch size gets updated by automatic memory optimization."""
        assert isinstance(self.instance, EvaluationTrainingCallback)
        with mock.patch.object(Evaluator, "evaluate", side_effect=self.instance.evaluator.evaluate) as mock_evaluate:
            pipeline(
                dataset="nations",
                model="distmult",
                training_kwargs=dict(
                    callbacks=self.instance,
                ),
            )
            assert {c.kwargs.get("batch_size", None) for c in mock_evaluate.call_args_list} != {None}
