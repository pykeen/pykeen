"""Tests for training callbacks."""

from collections.abc import MutableMapping
from typing import Any

from pykeen.training.callbacks import EvaluationLossTrainingCallback

from .. import cases


# TODO: more tests
class EvaluationLossTrainingCallbackTestCase(cases.TrainingCallbackTestCase):
    """Test for evaluation callback."""

    cls = EvaluationLossTrainingCallback

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["triples_factory"] = self.dataset.validation
        return kwargs
