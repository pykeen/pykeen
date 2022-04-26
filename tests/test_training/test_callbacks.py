"""Tests for training callbacks."""
from typing import Any, MutableMapping
from .. import cases
from pykeen.training.callbacks import EvaluationTrainingCallback


class EvaluationTrainingCallbackTestCase(cases.TrainingCallbackTestCase):
    """Test for evaluation callback."""

    cls = EvaluationTrainingCallback

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["evaluation_triples"] = self.dataset.validation.mapped_triples
        return kwargs
