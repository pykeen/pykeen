"""Test automatic batch size reduction."""

from collections.abc import Iterator

import pytest
import torch
from torch import FloatTensor

from pykeen import predict
from pykeen.datasets.nations import Nations
from pykeen.evaluation.evaluation_loop import LCWAEvaluationLoop
from pykeen.models.mocks import FixedModel
from pykeen.triples.triples_factory import CoreTriplesFactory


class MockModel(FixedModel):
    """A version of the FixedModel with simulates OOM."""

    def __init__(self, **kwargs):
        """Initialize the model."""
        super().__init__(**kwargs)
        self.oom_count = 0

    def _generate_fake_scores(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        # raise an OOM error whenever the batch size is larger than 1
        if torch.as_tensor([x.shape for x in (h, r, t)]).max(axis=0).values.min() > 1:
            self.oom_count += 1
            raise torch.cuda.OutOfMemoryError
        return super()._generate_fake_scores(h, r, t)


@pytest.fixture(scope="session")
def triples_factory() -> CoreTriplesFactory:
    """Return a fixture for an KG info."""
    return Nations().training


@pytest.fixture()
def model(triples_factory: CoreTriplesFactory) -> Iterator[MockModel]:
    """Return a fixture for a model."""
    model = MockModel(triples_factory=triples_factory)
    yield model
    assert model.oom_count


def test_evaluation_loop(model: MockModel, triples_factory: CoreTriplesFactory) -> None:
    """Test evaluation loop batch size reduction."""
    loop = LCWAEvaluationLoop(triples_factory=triples_factory, model=model)
    result = loop.evaluate(batch_size=None, use_tqdm=False)
    assert result


def test_predict_all(model: MockModel) -> None:
    """Test all triple prediction batch size reduction."""
    score_pack = predict.predict_all(model=model, k=16, batch_size=None)
    assert score_pack


def test_predict_triples(model: MockModel, triples_factory: CoreTriplesFactory) -> None:
    """Test triples scoring batch size reduction."""
    score_pack = predict.predict_triples(model=model, triples=triples_factory.mapped_triples)
    assert score_pack
