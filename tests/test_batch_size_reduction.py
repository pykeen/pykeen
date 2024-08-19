"""Test automatic batch size reduction."""

import pytest
import torch
from torch import FloatTensor

from pykeen.datasets.nations import Nations
from pykeen.evaluation.evaluation_loop import LCWAEvaluationLoop
from pykeen.models.base import Model
from pykeen.models.mocks import FixedModel
from pykeen.triples.triples_factory import CoreTriplesFactory


class MockModel(FixedModel):
    """A version of the FixedModel with simulates OOM."""

    def _generate_fake_scores(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        # raise an OOM error whenever the batch size is larger than 1
        if torch.as_tensor([x.shape for x in (h, r, t)]).max(axis=0).values.min() > 1:
            raise torch.cuda.OutOfMemoryError
        return super()._generate_fake_scores(h, r, t)


@pytest.fixture(scope="session")
def triples_factory() -> CoreTriplesFactory:
    """Return a fixture for an KG info."""
    return Nations().training


@pytest.fixture(scope="session")
def model(triples_factory: CoreTriplesFactory) -> Model:
    """Return a fixture for a model."""
    return MockModel(triples_factory=triples_factory)


def test_evaluation_loop(model: Model, triples_factory: CoreTriplesFactory):
    """Test evaluation loop batch size reduction."""
    loop = LCWAEvaluationLoop(triples_factory=triples_factory, model=model)
    result = loop.evaluate(batch_size=None, use_tqdm=False)
    assert result
