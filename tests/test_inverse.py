"""Tests for inverse relation handling."""

from collections.abc import Callable

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.mocks import create_inductive_dataset
from pykeen.inverse import DefaultRelationInverter, RelationInverter, relation_inverter_resolver
from pykeen.models import (
    CompGCN,
    ConvE,
    CooccurrenceFilteredModel,
    InductiveNodePiece,
    Model,
    NodePiece,
    TargetScoringBatch,
    TransE,
    TripleScoringBatch,
)
from pykeen.nn.representation import CombinedCompGCNRepresentations, SingleCompGCNRepresentation
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop, TrainingLoop
from pykeen.triples.instances import LCWAInstances
from pykeen.typing import TESTING, TRAINING, VALIDATION, InductiveMode, Target


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
    assert not torch.as_tensor(inverter.is_inverse(mapped[:, 1])).any()
    assert torch.as_tensor(inverter.is_inverse(inverter.to_internal_batch(batch=batch, invert=True)[:, 1])).all()


def _forward(
    model: Model, pairs: torch.Tensor, target: Target, ids: torch.Tensor | None = None, **kwargs
) -> torch.Tensor:
    """Score the target against all or the given candidates, given the pairs with *internal* relation IDs."""
    return model(TargetScoringBatch.from_pairs(pairs, target=target, target_ids=ids), **kwargs)


@pytest.mark.parametrize(
    ("method_name", "columns"),
    [("predict_hrt", [0, 1, 2]), ("predict_t", [0, 1]), ("predict_h", [1, 2])],
)
def test_predict_inverse_does_not_modify_input(model: Model, method_name: str, columns: list[int]):
    """Test that predicting by means of inverse triples does not modify the input batch."""
    batch = Nations().testing.mapped_triples[:3, columns].clone()
    copy = batch.clone()
    getattr(model, method_name)(batch, invert=True)
    assert torch.equal(batch, copy)


@pytest.mark.parametrize("use_inverse_triples", [False, True])
def test_predict_r_uses_real_relation_ids(use_inverse_triples: bool):
    """Test that predict_r scores against the real relations, while the scoring request stays internal."""
    factory = Nations().training
    model = TransE(triples_factory=factory, embedding_dim=2, random_seed=0, use_inverse_triples=use_inverse_triples)
    ht_batch = torch.as_tensor([[0, 1], [2, 3]])

    # scoring requests operate on the model's internal relations, which comprise the inverse ones
    assert _forward(model, ht_batch, "relation").shape[-1] == model.num_relations

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
def test_prepare_training_triples(flag: bool):
    """Test that the model flag controls whether inverse triples are added to the training triples."""
    tf = Nations().training
    model = TransE(triples_factory=tf, embedding_dim=2, random_seed=0, use_inverse_triples=flag)
    mapped_triples = model.prepare_training_triples(tf.mapped_triples)
    instances = LCWAInstances.from_triples_factory(tf, mapped_triples=mapped_triples, num_relations=model.num_relations)
    relations = torch.as_tensor(instances.pairs)[:, 1]
    if flag:
        assert len(mapped_triples) == 2 * len(tf.mapped_triples)
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
def test_model_builds_with_inverse_relations(model_cls: Callable[..., Model]):
    """Test that models requiring inverse relations enable them."""
    tf = Nations().training
    model = model_cls(triples_factory=tf, embedding_dim=16, random_seed=0)
    assert model.use_inverse_triples
    assert model.num_relations == 2 * tf.real_num_relations
    # scoring with inverse relations works
    model.predict_h(tf.mapped_triples[:4, 1:], invert=True)


@pytest.mark.parametrize("model_cls", [NodePiece, CompGCN])
def test_model_requires_flag(model_cls: Callable[..., Model]):
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
    inverse_target = batch.invert(relation_inverter)
    assert inverse_target.target == "tail"
    assert inverse_target.target_ids is not None
    assert torch.equal(inverse_target.target_ids, ids)
    assert inverse_target.indices.head is not None
    assert torch.equal(inverse_target.indices.head, tail)
    assert inverse_target.batch_shape == batch.batch_shape

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
    expected = _forward(model, tr_inv_batch, "tail", heads)
    torch.testing.assert_close(model.predict_h(rt_batch, heads=heads), expected)
    # ... which is the default for head prediction, and can be disabled
    torch.testing.assert_close(model.predict_h(rt_batch, heads=heads, invert=True), expected)
    forward = _forward(model, model.relation_inverter.to_internal_batch(batch=rt_batch, index=0), "head", heads)
    torch.testing.assert_close(model.predict_h(rt_batch, heads=heads, invert=False), forward)


def test_inverse_scoring_requires_flag():
    """Test that predicting by means of inverse triples raises an error for models without inverse relations."""
    model = TransE(triples_factory=Nations().training, embedding_dim=2, random_seed=0)
    with pytest.raises(ValueError, match="use_inverse_triples=True"):
        model.predict_h(Nations().testing.mapped_triples[:3, 1:], invert=True)


def test_predict_h_slice_and_per_batch_heads(model: Model):
    """Test that head prediction with per-batch heads and slicing matches scoring the tails of the inverse triples."""
    rt_batch = Nations().testing.mapped_triples[:3, 1:]
    heads = torch.randint(model.num_entities, size=(3, 4), generator=torch.manual_seed(0))
    tr_inv_batch = model.relation_inverter.to_internal_batch(batch=rt_batch, index=0, invert=True).flip(1)
    expected = _forward(model, tr_inv_batch, "tail", heads)
    torch.testing.assert_close(model.predict_h(rt_batch, heads=heads), expected)
    torch.testing.assert_close(model.predict_h(rt_batch, heads=heads, slice_size=2), expected)


def test_predict_t_inverse(model: Model):
    """Test that the tail scores via inverse relations equal the head scores of the manually inverted pairs."""
    hr_batch = Nations().testing.mapped_triples[:3, :2]
    hr_internal = model.relation_inverter.to_internal_batch(batch=hr_batch, index=1)
    r_inv = model.relation_inverter.get_inverse_id(hr_internal[:, 1])
    expected = _forward(model, torch.stack([r_inv, hr_internal[:, 0]], dim=-1), "head")
    torch.testing.assert_close(model.predict_t(hr_batch, invert=True), expected)


def test_subclass_score_is_used(model: Model):
    """Test that an overridden ``_score`` is used by ``predict_*``, with and without inverse triples."""

    class ShiftedTransE(TransE):
        """A model overriding the single scoring override point."""

        def _score(self, batch, **kwargs):
            return super()._score(batch, **kwargs) + 1.0

    shifted = ShiftedTransE(
        triples_factory=Nations().training, embedding_dim=2, random_seed=0, use_inverse_triples=True
    )
    triples = Nations().testing.mapped_triples[:3]
    hr_batch = triples[:, :2]
    torch.testing.assert_close(shifted.predict_t(hr_batch), model.predict_t(hr_batch) + 1.0)
    torch.testing.assert_close(shifted.predict_t(hr_batch, invert=True), model.predict_t(hr_batch, invert=True) + 1.0)


def test_predict_unknown_kwarg(model: Model):
    """Test that unknown keyword arguments are rejected."""
    with pytest.raises(TypeError):
        model.predict_t(Nations().testing.mapped_triples[:3, :2], foo=1)


@pytest.mark.parametrize("conjunctive", [False, True])
@pytest.mark.parametrize("use_inverse_triples", [False, True])
def test_filtered_model_training_mask(use_inverse_triples: bool, conjunctive: bool):
    """Test that the training-time co-occurrence mask is computed in the internal relation ID space."""
    tf = Nations().training
    model = CooccurrenceFilteredModel(
        triples_factory=tf,
        base=TransE,
        use_inverse_triples=use_inverse_triples,
        apply_in_training=True,
        conjunctive=conjunctive,
        training_fill_value=float("-inf"),
        random_seed=0,
    )
    # the triples as seen by the scoring methods, i.e., with internal relation IDs and including the inverse triples
    inverter = model.relation_inverter
    h, r, t = tf.mapped_triples.t()
    internal = tf.mapped_triples
    if use_inverse_triples:
        r = inverter.to_internal(r)
        internal = torch.cat([torch.stack([h, r, t], dim=-1), torch.stack([t, inverter.get_inverse_id(r), h], dim=-1)])
    # dense co-occurrence matrices; the mask is their elementwise conjunction / disjunction
    num_entities, num_relations = tf.num_entities, model.num_relations
    h_r, h_t, r_t = (
        torch.zeros(n, m, dtype=torch.bool)
        for n, m in ((num_entities, num_relations), (num_entities, num_entities), (num_relations, num_entities))
    )
    h_r[internal[:, 0], internal[:, 1]] = True
    h_t[internal[:, 0], internal[:, 2]] = True
    r_t[internal[:, 1], internal[:, 2]] = True

    def combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a & b) if conjunctive else (a | b)

    h, r, t = internal.t()
    for scores, expected in (
        (_forward(model, internal[:, 1:], "head"), combine(h_r.t()[r], h_t.t()[t])),
        (_forward(model, internal[:, 0::2], "relation"), combine(h_r[h], r_t.t()[t])),
        (_forward(model, internal[:, :2], "tail"), combine(h_t[h], r_t[r])),
    ):
        assert torch.equal(torch.isfinite(scores), expected)


@pytest.mark.parametrize("wrap_instance", [False, True])
@pytest.mark.parametrize("apply_in_training", [False, True])
@pytest.mark.parametrize("conjunctive", [False, True])
@pytest.mark.parametrize("mode", [TRAINING, VALIDATION, TESTING])
def test_filtered_model_inductive_mask(
    mode: InductiveMode, conjunctive: bool, apply_in_training: bool, wrap_instance: bool
):
    """Test that the co-occurrence mask is computed from the training graph, or the inference graph for evaluation.

    The base model is either instantiated by the filtered model, or an already instantiated model is wrapped.
    """
    # the inference graph has its own (fewer) entities than the training graph
    dataset = create_inductive_dataset(
        num_relations=3,
        num_entities_transductive=13,
        num_triples_training=33,
        num_entities_inductive=5,
        num_triples_inference=31,
        num_triples_testing=37,
    )
    training, inference = dataset.transductive_training, dataset.inductive_inference
    base_kwargs = {"embedding_dim": 2}
    base: InductiveNodePiece | type[InductiveNodePiece] = InductiveNodePiece
    if wrap_instance:
        base = InductiveNodePiece(triples_factory=training, inference_factory=inference, embedding_dim=2)
        base_kwargs = {}
    model = CooccurrenceFilteredModel(
        triples_factory=training,
        base=base,
        inference_factory=inference,
        **base_kwargs,
        apply_in_training=apply_in_training,
        conjunctive=conjunctive,
        training_fill_value=float("-inf"),
        random_seed=0,
    )
    # validation and testing triples are queries against the inference graph
    graph = dataset.transductive_training if mode == TRAINING else dataset.inductive_inference
    assert model._get_entity_len(mode=mode) == graph.num_entities
    triples = graph.mapped_triples
    # the scoring sees the internal relation IDs, including the inverse triples
    inverter = model.relation_inverter
    h, r, t = triples.t()
    r = inverter.to_internal(r)
    triples = torch.cat([torch.stack([h, r, t], dim=-1), torch.stack([t, inverter.get_inverse_id(r), h], dim=-1)])
    num_relations = model.num_relations
    # the masking is only applied in training mode if requested; otherwise, it is applied for predictions
    model.train(apply_in_training)

    def score_h(pairs, mode):
        return _forward(model, pairs, "head", mode=mode)

    def score_r(pairs, mode):
        return _forward(model, pairs, "relation", mode=mode)

    def score_t(pairs, mode):
        return _forward(model, pairs, "tail", mode=mode)

    # dense co-occurrence matrices; the mask is their elementwise conjunction / disjunction
    num_entities = graph.num_entities
    h_r, h_t, r_t = (
        torch.zeros(n, m, dtype=torch.bool)
        for n, m in ((num_entities, num_relations), (num_entities, num_entities), (num_relations, num_entities))
    )
    h_r[triples[:, 0], triples[:, 1]] = True
    h_t[triples[:, 0], triples[:, 2]] = True
    r_t[triples[:, 1], triples[:, 2]] = True

    def combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a & b) if conjunctive else (a | b)

    h, r, t = triples.t()
    for scores, expected in (
        (score_h(triples[:, 1:], mode=mode), combine(h_r.t()[r], h_t.t()[t])),
        (score_r(triples[:, 0::2], mode=mode), combine(h_r[h], r_t.t()[t])),
        (score_t(triples[:, :2], mode=mode), combine(h_t[h], r_t[r])),
    ):
        assert scores.shape == expected.shape
        assert torch.equal(torch.isfinite(scores), expected)


@pytest.mark.parametrize("mode", [VALIDATION, TESTING])
def test_filtered_model_requires_inference_factory(mode: InductiveMode):
    """Test that the filtered model rejects an inference mode without an inference graph."""
    model = CooccurrenceFilteredModel(triples_factory=Nations().training, base=TransE, random_seed=0)
    with pytest.raises(ValueError, match="inference_factory"):
        model._get_indexes(mode=mode)


class _SwappedRelationInverter(DefaultRelationInverter):
    """A non-default relation inverter, which uses odd IDs for forward relations and even IDs for inverse ones."""

    def to_internal(self, relation_id):
        return (relation_id << 1) | 1

    def is_inverse(self, relation_id):
        return (relation_id & 1) == 0


def test_compgcn_uses_relation_inverter():
    """Test that the CompGCN encoder picks forward and inverse relation representations via the relation inverter."""
    tf = Nations().training
    default, swapped = (
        CombinedCompGCNRepresentations(
            triples_factory=tf,
            entity_representations_kwargs={"embedding_dim": 4},
            relation_representations_kwargs={"embedding_dim": 4},
            relation_inverter=relation_inverter,
        ).eval()
        for relation_inverter in (None, _SwappedRelationInverter())
    )
    assert type(default.relation_inverter) is DefaultRelationInverter
    # use the same parameters, except for exchanging the forward and inverse relation representations
    state = default.state_dict()
    key = "relation_representations._embeddings.weight"
    permutation = torch.arange(state[key].shape[0]) ^ 1
    state[key] = state[key][permutation]
    swapped.load_state_dict(state)
    with torch.no_grad():
        x_e_default, x_r_default = default()
        x_e_swapped, x_r_swapped = swapped()
    assert torch.allclose(x_e_default, x_e_swapped)
    assert torch.allclose(x_r_default, x_r_swapped[permutation])


def test_compgcn_encoder_shares_model_relation_inverter():
    """Test that the CompGCN model's encoder uses the model's relation inverter."""
    model: CompGCN = CompGCN(triples_factory=Nations().training, embedding_dim=4, random_seed=0)
    representation = model.entity_representations[0]
    assert isinstance(representation, SingleCompGCNRepresentation)
    assert representation.combined.relation_inverter is model.relation_inverter


def test_compgcn_rejects_encoder_relation_inverter():
    """Test that CompGCN rejects a relation inverter in the encoder arguments, and uses the model's one."""
    tf = Nations().training
    with pytest.raises(ValueError, match="encoder_kwargs"):
        CompGCN(
            triples_factory=tf,
            embedding_dim=4,
            random_seed=0,
            encoder_kwargs={"relation_inverter": _SwappedRelationInverter()},
        )
    inverter = _SwappedRelationInverter()
    model: CompGCN = CompGCN(triples_factory=tf, embedding_dim=4, random_seed=0, relation_inverter=inverter)
    assert model.relation_inverter is inverter


def test_relation_inverter_equality():
    """Test that relation inverters are compared by type."""
    assert DefaultRelationInverter() == DefaultRelationInverter()
    assert hash(DefaultRelationInverter()) == hash(DefaultRelationInverter())
    assert DefaultRelationInverter() != _SwappedRelationInverter()
    assert DefaultRelationInverter() != "default"
