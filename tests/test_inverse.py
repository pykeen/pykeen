"""Tests for inverse relation handling."""

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.inverse import DefaultRelationInverter, RelationInverter, relation_inverter_resolver
from pykeen.models import Model, TransE


@pytest.fixture(params=relation_inverter_resolver.lookup_dict.values())
def relation_inverter(request) -> RelationInverter:
    """Return a relation inverter."""
    return request.param()


@pytest.fixture
def model() -> Model:
    """Return a model trained with inverse relations."""
    return TransE(triples_factory=Nations(create_inverse_triples=True).training, embedding_dim=2, random_seed=0)


def test_invert_does_not_modify_input(relation_inverter: RelationInverter):
    """Test that the non-inplace variant leaves its input alone."""
    batch = torch.as_tensor([[0, 1, 2], [3, 4, 5]])
    copy = batch.clone()
    inverted = relation_inverter.invert(batch=batch)
    assert torch.equal(batch, copy)
    assert not torch.equal(inverted, copy)
    # the in-place variant applied to a copy has to agree
    assert torch.equal(inverted, relation_inverter.invert_(batch=copy))


def test_default_inverter_ids():
    """Test the ID scheme of the default relation inverter."""
    inverter = DefaultRelationInverter()
    batch = torch.as_tensor([[0, 3, 1]])
    mapped = inverter.map(batch=batch)
    assert mapped[0, 1].item() == 6
    assert inverter.get_inverse_id(relation_id=mapped[0, 1]).item() == 7
    assert not inverter.is_inverse(mapped[:, 1]).any()
    assert inverter.is_inverse(inverter.map(batch=batch, invert=True)[:, 1]).all()


@pytest.mark.parametrize(
    ("method_name", "columns"),
    [("score_hrt_inverse", [0, 1, 2]), ("score_t_inverse", [0, 1]), ("score_h_inverse", [1, 2])],
)
def test_score_inverse_does_not_modify_input(model: Model, method_name: str, columns: list[int]):
    """Test that the public inverse scoring methods do not modify their input batch."""
    batch = Nations().testing.mapped_triples[:3, columns].clone()
    copy = batch.clone()
    getattr(model, method_name)(batch)
    assert torch.equal(batch, copy)
