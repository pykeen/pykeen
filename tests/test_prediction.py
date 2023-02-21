"""Tests for prediction tools."""
from typing import Any, Collection, Iterable, MutableMapping, Optional, Sequence, Tuple, Union

import numpy
import pandas
import pytest
import torch
import unittest_templates

import pykeen.models
import pykeen.models.mocks
import pykeen.predict
import pykeen.regularizers
import pykeen.typing
from pykeen.constants import COLUMN_LABELS
from pykeen.datasets.nations import Nations
from pykeen.triples.triples_factory import AnyTriples, CoreTriplesFactory, KGInfo
from pykeen.utils import resolve_device
from tests import cases


@pytest.mark.parametrize("size", [(10,), (10, 3)])
def test_isin_many_dim(size: Tuple[int, ...]):
    """Tests for isin_many_dim."""
    generator = torch.manual_seed(seed=42)
    elements = torch.rand(size=size, generator=generator)
    test_elements = torch.cat([elements[:3], torch.rand(size=size, generator=generator)])
    mask = pykeen.predict.isin_many_dim(elements=elements, test_elements=test_elements, dim=0)
    assert mask.shape == (elements.shape[0],)
    assert mask.dtype == torch.bool
    assert mask[:3].all()


# prediction post-processing
class TargetPredictionsTests(cases.PredictionTestCase):
    """Tests for target prediction post-processing."""

    cls = pykeen.predict.TargetPredictions
    kwargs = dict(target=pykeen.typing.LABEL_HEAD, other_columns_fixed_ids=(0, 1))

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        generator = torch.manual_seed(seed=42)
        target = kwargs["target"]
        self.df = kwargs["df"] = pandas.DataFrame(
            data={
                f"{target}_id": range(self.dataset.num_entities),
                "score": torch.rand(size=(self.dataset.num_entities,), generator=generator),
            }
        )
        return kwargs


class TriplePredictionsTest(cases.PredictionTestCase):
    """Tests for triple scoring post-processing."""

    cls = pykeen.predict.TriplePredictions

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        # mock prediction data frame
        generator = torch.manual_seed(seed=42)
        data = {
            f"{label}_id": torch.randint(max_id, size=(5,), generator=generator).numpy()
            for label, max_id in zip(
                COLUMN_LABELS, [self.dataset.num_entities, self.dataset.num_relations, self.dataset.num_entities]
            )
        }
        data["score"] = torch.rand(size=(5,), generator=generator).numpy()
        self.df = kwargs["df"] = pandas.DataFrame(data=data)
        return kwargs


class PredictionsMetaTestCase(unittest_templates.MetaTestCase[pykeen.predict.Predictions]):
    """Test for tests for prediction post processing."""

    base_cls = pykeen.predict.Predictions
    base_test = cases.PredictionTestCase


# score consumers


class CountScoreConsumerTestCase(cases.ScoreConsumerTests):
    """Test count score consumer."""

    cls = pykeen.predict.CountScoreConsumer


class TopKScoreConsumerTestCase(cases.ScoreConsumerTests):
    """Test top-k score consumer."""

    cls = pykeen.predict.TopKScoreConsumer


class AllScoreConsumerTestCase(cases.ScoreConsumerTests):
    """Test all score consumer."""

    cls = pykeen.predict.AllScoreConsumer
    kwargs = dict(
        num_entities=cases.ScoreConsumerTests.num_entities,
        num_relations=cases.ScoreConsumerTests.num_entities,
    )


class ScoreConsumerMetaTestCase(unittest_templates.MetaTestCase[pykeen.predict.ScoreConsumer]):
    """Test for tests for score consumers."""

    base_cls = pykeen.predict.ScoreConsumer
    base_test = cases.ScoreConsumerTests


@pytest.mark.parametrize(["num_entities", "num_relations"], [(3, 2)])
def test_consume_scores(num_entities: int, num_relations: int):
    """Test for consume_scores."""
    dataset = pykeen.predict.AllPredictionDataset(num_entities=num_entities, num_relations=num_relations)
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=False)
    )
    consumer = pykeen.predict.CountScoreConsumer()
    pykeen.predict.consume_scores(model, dataset, consumer)
    assert consumer.batch_count == num_relations * num_entities
    assert consumer.score_count == num_relations * num_entities**2


def _iter_predict_all_inputs() -> Iterable[Tuple[pykeen.models.Model, Optional[int], pykeen.typing.Target, int]]:
    """Iterate over test inputs for predict_all."""
    # use a small model, since operation is expensive
    num_entities, num_relations = 3, 2
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=False)
    )
    # all scores, automatic batch size
    yield model, None, pykeen.typing.LABEL_TAIL, None
    # top 3 scores
    yield model, 3, pykeen.typing.LABEL_TAIL, None
    # top 3 scores, fixed batch size, head scoring
    yield model, 3, pykeen.typing.LABEL_HEAD, 2
    # all scores, relation scoring
    yield model, 3, pykeen.typing.LABEL_RELATION, None
    # all scores, relation scoring
    yield model, 3, pykeen.typing.LABEL_RELATION, None
    # model with inverse relations
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=True)
    )
    yield model, None, pykeen.typing.LABEL_TAIL, None


def _check_score_pack(pack: pykeen.predict.ScorePack, model: pykeen.models.Model, num_triples: int):
    """Check store pack properties."""
    assert isinstance(pack, pykeen.predict.ScorePack)
    # check type
    assert isinstance(pack.result, torch.Tensor)
    assert pack.result.dtype == torch.long
    assert isinstance(pack.scores, torch.Tensor)
    assert pack.scores.is_floating_point()
    # check shape
    assert pack.result.shape == (num_triples, 3)
    assert pack.scores.shape == (num_triples,)
    # check ID ranges
    assert (pack.result >= 0).all()
    assert pack.result[:, [0, 2]].max() < model.num_entities
    assert pack.result[:, 1].max() < model.num_relations


@pytest.mark.parametrize(["model", "k", "target", "batch_size"], _iter_predict_all_inputs())
def test_predict_all(model: pykeen.models.Model, k: Optional[int], target: pykeen.typing.Target, batch_size: int):
    """Test the predict method."""
    pack = pykeen.predict.predict_all(model=model, k=k, target=target, batch_size=batch_size)
    _check_score_pack(
        pack=pack, model=model, num_triples=model.num_entities**2 * model.num_relations if k is None else k
    )


def test_predict_top_k_consistency():
    """Test consistency of top-k scoring."""
    ks = [5, 10]
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=3, num_relations=5, create_inverse_triples=False)
    )
    dfs = [
        pykeen.predict.predict_all(model=model, k=k)
        .process()
        .df.nlargest(n=min(ks), columns="score")
        .reset_index(drop=True)
        for k in ks
    ]
    assert set(dfs[0].columns) == set(dfs[0].columns)
    for column in dfs[0].columns:
        numpy.testing.assert_equal(dfs[0][column].values, dfs[1][column].values)


def _iter_predict_triples_inputs() -> (
    Iterable[Tuple[pykeen.models.Model, AnyTriples, Optional[CoreTriplesFactory], Optional[int]]]
):
    """Iterate over test inputs for predict_triples."""
    dataset = Nations()
    factory = dataset.training
    model = pykeen.models.mocks.FixedModel(triples_factory=factory)
    mapped_triples = factory.mapped_triples[:3]
    # mapped triples, automatic batch size selection, no factory
    yield model, mapped_triples, None, None
    # mapped triples, fixed batch size, no factory
    yield model, mapped_triples, None, 2
    # labeled triples with factory
    labeled = factory.label_triples(mapped_triples)
    yield model, labeled, factory, None
    # labeled triples as list
    labeled_list = labeled.tolist()
    yield model, labeled_list, factory, None
    # single labeled triple
    yield model, labeled_list[0], factory, None
    # model with inverse relations
    dataset = Nations(create_inverse_triples=True)
    factory = dataset.training
    model = pykeen.models.mocks.FixedModel(triples_factory=factory)
    yield model, factory.mapped_triples[:3], None, None


@pytest.mark.parametrize(["model", "triples", "triples_factory", "batch_size"], _iter_predict_triples_inputs())
def test_predict_triples(
    model: pykeen.models.Model,
    triples: AnyTriples,
    triples_factory: Optional[CoreTriplesFactory],
    batch_size: Optional[int],
):
    """Test triple scoring."""
    pack = pykeen.predict.predict_triples(
        model=model, triples=triples, triples_factory=triples_factory, batch_size=batch_size
    )
    if not isinstance(triples, (torch.Tensor, numpy.ndarray)) and isinstance(triples[0], str):
        num_triples = 1
    else:
        num_triples = len(triples)
    _check_score_pack(pack=pack, model=model, num_triples=num_triples)


def _iter_get_input_batch_inputs() -> (
    Iterable[
        Tuple[
            Optional[CoreTriplesFactory],
            Union[None, int, str],
            Union[None, int, str],
            Union[None, int, str],
            pykeen.typing.Target,
        ]
    ]
):
    """Iterate over test inputs for _get_input_batch."""
    factory = Nations().training
    # ID-based, no factory
    yield None, 0, 1, None, pykeen.typing.LABEL_TAIL
    yield None, 1, None, 0, pykeen.typing.LABEL_RELATION
    yield None, None, 0, 1, pykeen.typing.LABEL_HEAD
    # string-based + factory
    yield factory, "uk", "accusation", None, pykeen.typing.LABEL_TAIL
    yield factory, "uk", None, "uk", pykeen.typing.LABEL_RELATION
    yield factory, None, "accusation", "uk", pykeen.typing.LABEL_HEAD
    # mixed + factory
    yield factory, 0, "accusation", None, pykeen.typing.LABEL_TAIL
    yield factory, "uk", None, 0, pykeen.typing.LABEL_RELATION
    yield factory, None, 1, "uk", pykeen.typing.LABEL_HEAD


@pytest.mark.parametrize(["factory", "head", "relation", "tail", "exp_target"], _iter_get_input_batch_inputs())
def test_get_input_batch(
    factory: Optional[CoreTriplesFactory],
    head: Union[None, int, str],
    relation: Union[None, int, str],
    tail: Union[None, int, str],
    exp_target: pykeen.typing.Target,
):
    """Test input batch construction for target prediction."""
    target, batch, batch_tuple = pykeen.predict._get_input_batch(
        factory=factory, head=head, relation=relation, tail=tail
    )
    assert target == exp_target
    assert isinstance(batch, torch.Tensor)
    assert isinstance(batch_tuple, tuple)
    assert batch.shape == (1, 2)
    assert len(batch_tuple) == 2
    assert batch.flatten().tolist() == list(batch_tuple)


def _iter_get_targets_inputs() -> (
    Iterable[Tuple[Union[None, torch.Tensor, Collection[Union[str, int]]], Optional[CoreTriplesFactory], bool]]
):
    """Iterate over test inputs for _get_targets."""
    factory = Nations().training
    for entity, id_to_label in ((True, factory.entity_id_to_label), (False, factory.relation_id_to_label)):
        all_labels = [label for _, label in sorted(id_to_label.items())]
        # no restriction, no factory
        yield None, None, entity, None, None, None
        # no restriction, factory
        yield None, factory, entity, all_labels, None, None
        # id restriction, no factory ...
        id_list = [0, 2, 3]
        id_tensor = torch.as_tensor(id_list, dtype=torch.long)
        yield id_list, None, entity, None, id_list, id_tensor
        yield id_tensor, None, entity, None, id_list, id_tensor
        # id restriction with factory
        labels = [all_labels[i] for i in id_list]
        yield id_list, factory, entity, labels, id_list, id_tensor
        yield id_tensor, factory, entity, labels, id_list, id_tensor


@pytest.mark.parametrize(
    ["ids", "factory", "entity", "exp_labels", "exp_ids", "exp_tensor"], _iter_get_targets_inputs()
)
def test_get_targets(
    ids: Union[None, torch.Tensor, Collection[Union[str, int]]],
    factory: Optional[CoreTriplesFactory],
    entity: bool,
    exp_labels: Optional[Sequence[str]],
    exp_ids: Optional[Sequence[int]],
    exp_tensor: Optional[torch.Tensor],
):
    """Test target normalization for target prediction."""
    device = resolve_device(device=None)
    labels_list, ids_list, ids_tensor = pykeen.predict._get_targets(
        ids=ids, triples_factory=factory, device=device, entity=entity
    )
    if exp_labels is None:
        assert labels_list is None
    else:
        assert list(labels_list) == list(exp_labels)
    if exp_ids is None:
        assert exp_ids is None
    else:
        assert list(ids_list) == list(exp_ids)
    if exp_tensor is None:
        assert ids_tensor is None
    else:
        assert (ids_tensor == exp_tensor).all()


def _iter_predict_target_inputs() -> (
    Iterable[Tuple[pykeen.models.Model, int, int, int, Optional[CoreTriplesFactory], Optional[Sequence[int]]]]
):
    # comment: we only use id-based input, since the normalization has already been tested
    # create model
    factory = Nations().training
    model = pykeen.models.mocks.FixedModel(triples_factory=factory)
    for factory_ in (None, factory):
        # id-based head/relation/tail prediction, no restriction
        yield model, None, 0, 1, factory_, None
        yield model, 1, None, 2, factory_, None
        yield model, 0, 1, None, factory_, None
        # restriction by list of ints
        yield model, 0, 1, None, factory_, [0, 3, 7]


@pytest.mark.parametrize(["model", "head", "relation", "tail", "factory", "targets"], _iter_predict_target_inputs())
def test_predict_target(
    model: pykeen.models.Model,
    head: Union[None, int, str],
    relation: Union[None, int, str],
    tail: Union[None, int, str],
    factory: Optional[CoreTriplesFactory],
    targets: Union[None, torch.LongTensor, Sequence[Union[int, str]]],
):
    """Test target scoring."""
    pred = pykeen.predict.predict_target(
        model=model, head=head, relation=relation, tail=tail, triples_factory=factory, targets=targets
    )
    assert isinstance(pred, pykeen.predict.TargetPredictions)
    assert pred.factory == factory


@pytest.mark.parametrize(
    ["heads", "relations", "tails", "target"],
    [
        # tail prediction
        ([1, 2], [3], None, pykeen.typing.LABEL_TAIL),
    ],
)
def test_partially_restricted_prediction_dataset(heads, relations, tails, target):
    """Test for PartiallyRestrictedPredictionDataset."""
    ds = pykeen.predict.PartiallyRestrictedPredictionDataset(
        heads=heads, relations=relations, tails=tails, target=target
    )
    # try accessing each element
    for i in range(len(ds)):
        _ = ds[i]
