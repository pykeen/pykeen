"""Tests for inverse relation handling."""

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.inverse import DefaultRelationInverter, RelationInverter, relation_inverter_resolver
from pykeen.models import (
    CompGCN,
    ConvE,
    CooccurrenceFilteredModel,
    Model,
    NodePiece,
    TargetScoringBatch,
    TransE,
    TripleScoringBatch,
)
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop, TrainingLoop
from pykeen.triples.instances import LCWAInstances


@pytest.fixture(params=relation_inverter_resolver.lookup_dict.values())
def relation_inverter(request) -> RelationInverter:
    """Return a relation inverter."""
    return request.param()


@pytest.fixture
def model() -> Model:
    """Return a model trained with inverse relations."""
    return TransE(triples_factory=Nations().training, embedding_dim=2, random_seed=0, use_inverse_triples=True)


@pytest.mark.parametrize("method_name", ["invert_internal_batch", "to_internal_batch"])
def test_batch_methods_do_not_modify_input(relation_inverter: RelationInverter, method_name: str):
    """Test that the batch-level methods leave their input alone."""
    batch = torch.as_tensor([[0, 1, 2], [3, 4, 5]])
    copy = batch.clone()
    result = getattr(relation_inverter, method_name)(batch=batch)
    assert torch.equal(batch, copy)
    assert not torch.equal(result, copy)


def test_get_inverse_id_is_involution(relation_inverter: RelationInverter):
    """Test that applying the inverse-ID mapping twice is the identity."""
    relation_id = torch.arange(10)
    inverse_id = relation_inverter.get_inverse_id(relation_id=relation_id)
    # an inverse relation's inverse is the forward relation again
    assert torch.equal(relation_inverter.get_inverse_id(relation_id=inverse_id), relation_id)
    # ... and forward and inverse IDs are never the same
    assert not torch.equal(inverse_id, relation_id)
    assert relation_inverter.is_inverse(inverse_id).sum() == relation_id.numel() // 2


def test_is_inverse_accepts_plain_ints(relation_inverter: RelationInverter):
    """Test that the ID-level methods work on plain integers, too."""
    for relation in range(5):
        internal = relation_inverter.to_internal(relation_id=relation)
        assert not relation_inverter.is_inverse(relation_id=internal)
        assert relation_inverter.is_inverse(relation_id=relation_inverter.get_inverse_id(relation_id=internal))
        assert relation_inverter.to_real(relation_id=internal) == relation


def test_default_inverter_ids():
    """Test the ID scheme of the default relation inverter."""
    inverter = DefaultRelationInverter()
    batch = torch.as_tensor([[0, 3, 1]])
    mapped = inverter.to_internal_batch(batch=batch)
    assert mapped[0, 1].item() == 6
    assert inverter.get_inverse_id(relation_id=mapped[0, 1]).item() == 7
    assert not inverter.is_inverse(mapped[:, 1]).any()
    assert inverter.is_inverse(inverter.to_internal_batch(batch=batch, invert=True)[:, 1]).all()


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


@pytest.mark.parametrize("use_inverse_triples", [False, True])
def test_predict_r_uses_real_relation_ids(use_inverse_triples: bool):
    """Test that predict_r scores against the real relations, while score_r stays internal."""
    factory = Nations().training
    model = TransE(triples_factory=factory, embedding_dim=2, random_seed=0, use_inverse_triples=use_inverse_triples)
    ht_batch = torch.as_tensor([[0, 1], [2, 3]])

    # score_r operates on the model's internal relations, which comprise the inverse ones
    assert model.score_r(ht_batch).shape[-1] == model.num_relations

    # predict_r drops them again, so that the columns are indexed by the factory's relation IDs ...
    scores = model.predict_r(ht_batch)
    assert scores.shape[-1] == factory.real_num_relations
    # ... in the same order as when they are requested explicitly
    explicit = model.predict_r(ht_batch, relations=torch.arange(factory.real_num_relations))
    assert torch.allclose(scores, explicit)


def test_get_inverse_relation_id():
    """Test that the factory's inverse relation ID matches the one used by models."""
    factory = Nations().training
    model = TransE(triples_factory=factory, embedding_dim=2, random_seed=0, use_inverse_triples=True)
    for relation in range(factory.real_num_relations):
        inverse_id = factory.get_inverse_relation_id(relation)
        assert 0 <= inverse_id < model.num_relations
        assert model.relation_inverter.is_inverse(torch.as_tensor([inverse_id])).item()
        # this is the relation ID a model actually uses for the inverse of ``relation``
        batch = TripleScoringBatch.from_batch(torch.as_tensor([[0, relation, 1]]))
        expected = batch.to_internal(model.relation_inverter).invert(model.relation_inverter)
        assert expected.indices.relation.item() == inverse_id


def test_get_inverse_relation_id_errors():
    """Test the input validation of the factory's inverse relation ID lookup."""
    factory = Nations().training
    with pytest.raises(ValueError, match="Invalid relation"):
        factory.get_inverse_relation_id(factory.real_num_relations)
    with pytest.raises(ValueError, match="Invalid relation"):
        factory.get_inverse_relation_id(-1)


@pytest.mark.parametrize("model_flag", [None, False, True])
def test_model_flag(model_flag: bool | None):
    """Test that the model flag determines the number of relations, and that it is disabled by default."""
    tf = Nations().training
    kwargs = {} if model_flag is None else {"use_inverse_triples": model_flag}
    model = TransE(triples_factory=tf, embedding_dim=2, random_seed=0, **kwargs)
    expected = bool(model_flag)
    assert model.use_inverse_triples == expected
    assert model.num_relations == (2 if expected else 1) * tf.real_num_relations
    assert model.num_real_relations == tf.real_num_relations
    assert model.relation_representations[0].max_id == model.num_relations


@pytest.mark.parametrize("flag", [False, True])
def test_lcwa_instances_flag(flag: bool):
    """Test that the LCWA instances flag controls whether inverse triples are added."""
    tf = Nations().training
    instances = LCWAInstances.from_triples_factory(tf, create_inverse_triples=flag)
    relations = torch.as_tensor(instances.pairs)[:, 1]
    if flag:
        # internal relation IDs, comprising both, forward and inverse relations
        assert relations.max().item() < 2 * tf.real_num_relations
        is_inverse = tf.relation_inverter.is_inverse(relations)
        assert is_inverse.any()
        assert not is_inverse.all()
    else:
        # real relation IDs
        assert torch.equal(relations.unique(), tf.mapped_triples[:, 1].unique())


@pytest.mark.parametrize("training_loop_cls", [SLCWATrainingLoop, LCWATrainingLoop])
def test_training_with_model_flag(training_loop_cls: type[TrainingLoop]):
    """Test that training a model with inverse relations updates the inverse relation representations, too."""
    tf = Nations().training
    model = TransE(triples_factory=tf, embedding_dim=2, random_seed=0, use_inverse_triples=True)
    relation_representations = model.relation_representations[0]
    before = relation_representations().detach().clone()
    losses = training_loop_cls(model=model, triples_factory=tf).train(
        triples_factory=tf, num_epochs=2, batch_size=64, use_tqdm=False
    )
    assert losses
    assert torch.isfinite(torch.as_tensor(losses)).all()
    after = relation_representations().detach()
    is_inverse = model.relation_inverter.is_inverse(torch.arange(model.num_relations))
    # both, forward and inverse relation representations receive updates
    assert not torch.allclose(before[~is_inverse], after[~is_inverse])
    assert not torch.allclose(before[is_inverse], after[is_inverse])


@pytest.mark.parametrize("model_cls", [ConvE, NodePiece, CompGCN])
def test_model_builds_with_inverse_relations(model_cls: type[Model]):
    """Test that models requiring inverse relations enable them."""
    tf = Nations().training
    model = model_cls(triples_factory=tf, embedding_dim=16, random_seed=0)
    assert model.use_inverse_triples
    assert model.num_relations == 2 * tf.real_num_relations
    # scoring with inverse relations works
    model.score_h_inverse(rt_batch=tf.mapped_triples[:4, 1:])


@pytest.mark.parametrize("model_cls", [NodePiece, CompGCN])
def test_model_requires_flag(model_cls: type[Model]):
    """Test that models requiring inverse relations raise an error if they are disabled."""
    with pytest.raises(ValueError, match="use_inverse_triples=True"):
        model_cls(
            triples_factory=Nations().training,
            embedding_dim=4,
            random_seed=0,
            use_inverse_triples=False,
        )


def test_filtered_model_mirrors_base_flag():
    """Test that the co-occurrence filtered model uses the base model's flag."""
    tf = Nations().training
    model = CooccurrenceFilteredModel(triples_factory=tf, base=TransE, use_inverse_triples=True, random_seed=0)
    assert model.use_inverse_triples
    assert model.base.use_inverse_triples
    assert model.num_relations == model.base.num_relations


def test_scoring_batch_invert(relation_inverter: RelationInverter):
    """Test that inverting a scoring request swaps head and tail, and inverts the relations."""
    head, relation, tail = torch.as_tensor([0, 1]), torch.as_tensor([2, 5]), torch.as_tensor([3, 4])
    triples = TripleScoringBatch.from_transposed_batch(head=head, relation=relation, tail=tail)
    inverse = triples.invert(relation_inverter)
    assert torch.equal(inverse.indices.head, tail)
    assert torch.equal(inverse.indices.relation, relation_inverter.get_inverse_id(relation))
    assert torch.equal(inverse.indices.tail, head)
    # inverting twice is the identity
    assert all(map(torch.equal, inverse.invert(relation_inverter).indices, triples.indices))

    # predicting heads of (*, r, t) becomes predicting tails of (t, r_inv, *), for the same candidates
    ids = torch.as_tensor([6, 7, 8])
    batch = TargetScoringBatch.from_transposed_batch(head=ids, relation=relation, tail=tail, target="head")
    inverse = batch.invert(relation_inverter)
    assert inverse.target == "tail"
    assert torch.equal(inverse.target_ids, ids)
    assert torch.equal(inverse.indices.head, tail)
    assert inverse.batch_shape == batch.batch_shape

    # scoring against all relations cannot be inverted or converted, since the candidates' order is undefined
    batch = TargetScoringBatch.from_transposed_batch(head=head, relation=None, tail=tail, target="relation")
    for method in (batch.invert, batch.to_internal):
        with pytest.raises(ValueError, match="materialize"):
            method(relation_inverter)


def test_predict_h_via_inverse(model: Model):
    """Test that head prediction scores the tails of the inverse triples, including a heads restriction."""
    rt_batch = Nations().testing.mapped_triples[:3, 1:]
    heads = torch.as_tensor([0, 2, 4])
    # the inverse triples, (t, r_inv, *), with internal relation IDs
    tr_inv_batch = model.relation_inverter.to_internal_batch(batch=rt_batch, index=0, invert=True).flip(1)
    expected = model.score_t(tr_inv_batch, tails=heads)
    torch.testing.assert_close(model.predict_h(rt_batch, heads=heads), expected)
    # the inverse scoring method operates on internal relation IDs
    rt_internal = model.relation_inverter.to_internal_batch(batch=rt_batch, index=0)
    torch.testing.assert_close(model.score_h_inverse(rt_internal, heads=heads), expected)


def test_inverse_scoring_requires_flag():
    """Test that the inverse scoring methods raise an error for models without inverse relations."""
    model = TransE(triples_factory=Nations().training, embedding_dim=2, random_seed=0)
    with pytest.raises(ValueError, match="use_inverse_triples=True"):
        model.score_h_inverse(rt_batch=Nations().testing.mapped_triples[:3, 1:])
