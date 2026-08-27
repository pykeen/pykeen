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


def test_get_inverse_relation_id():
    """Test that the factory's inverse relation ID matches the one used by models."""
    factory = Nations(create_inverse_triples=True).training
    model = TransE(triples_factory=factory, embedding_dim=2, random_seed=0)
    for relation in range(factory.real_num_relations):
        inverse_id = factory.get_inverse_relation_id(relation)
        assert 0 <= inverse_id < factory.num_relations
        assert model.relation_inverter.is_inverse(torch.as_tensor([inverse_id])).item()
        # this is the relation ID a model actually uses for the inverse of ``relation``
        batch = torch.as_tensor([[0, relation, 1]])
        expected = model._prepare_inverse_batch(model._prepare_batch(batch, index_relation=1), index_relation=1)
        assert expected[0, 1].item() == inverse_id


def test_get_inverse_relation_id_errors():
    """Test the input validation of the factory's inverse relation ID lookup."""
    factory = Nations(create_inverse_triples=True).training
    with pytest.raises(ValueError, match="Invalid relation"):
        factory.get_inverse_relation_id(factory.real_num_relations)
    with pytest.raises(ValueError, match="they have not been created"):
        Nations().training.get_inverse_relation_id(0)


@pytest.mark.parametrize("create_inverse_triples", [False, True])
def test_create_inverse_triples_setter(create_inverse_triples: bool):
    """Test that toggling the flag keeps the number of relations consistent."""
    factory = Nations(create_inverse_triples=create_inverse_triples).training
    real_num_relations = factory.real_num_relations
    for flag in (True, False, True, create_inverse_triples):
        factory.create_inverse_triples = flag
        assert factory.create_inverse_triples == flag
        assert factory.real_num_relations == real_num_relations
        assert factory.num_relations == (2 * real_num_relations if flag else real_num_relations)
        # the (possibly inverted) triples have to stay within the ID range
        mapped_triples = factory._add_inverse_triples_if_necessary(mapped_triples=factory.mapped_triples)
        assert mapped_triples[:, 1].max().item() < factory.num_relations
