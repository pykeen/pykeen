"""Tests for prediction tools."""
import functools
import itertools
from typing import Any, MutableMapping, Optional, Tuple

import pandas
import pytest
import torch
import unittest_templates

import pykeen.models.mocks
import pykeen.models.predict
import pykeen.typing
from pykeen.triples.triples_factory import KGInfo
from tests import cases


@pytest.mark.parametrize("size", [(10,), (10, 3)])
def test_isin_many_dim(size: Tuple[int, ...]):
    """Tests for isin_many_dim."""
    generator = torch.manual_seed(seed=42)
    elements = torch.rand(size=size, generator=generator)
    test_elements = torch.cat([elements[:3], torch.rand(size=size, generator=generator)])
    mask = pykeen.models.predict.isin_many_dim(elements=elements, test_elements=test_elements, dim=0)
    assert mask.shape == (elements.shape[0],)
    assert mask.dtype == torch.bool
    assert mask[:3].all()


class SinglePredictionPostProcessorTest(cases.PredictionPostProcessorTestCase):
    """Tests for SinglePredictionPostProcessor."""

    cls = pykeen.models.predict.SinglePredictionPostProcessor

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        # mock prediction data frame
        kwargs["target"] = target = pykeen.typing.LABEL_TAIL
        labels, ids = list(zip(*self.dataset.entity_to_id.items()))
        score = torch.rand(len(ids), generator=torch.manual_seed(seed=42))
        self.df = pandas.DataFrame({f"{target}_id": ids, f"{target}_label": labels, "score": score})
        # set other parameters
        kwargs["other_columns_fixed_ids"] = tuple(self.dataset.training.mapped_triples[0, :2])
        return kwargs


class AllPredictionPostProcessorTest(cases.PredictionPostProcessorTestCase):
    """Tests for AllPredictionPostProcessor."""

    cls = pykeen.models.predict.AllPredictionPostProcessor

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        # mock prediction data frame
        data = list(
            map(
                functools.partial(sum, start=tuple()),
                itertools.product(
                    self.dataset.entity_to_id.items(),
                    self.dataset.relation_to_id.items(),
                    self.dataset.entity_to_id.items(),
                ),
            )
        )
        columns = list(
            itertools.chain.from_iterable(
                (f"{col}_label", f"{col}_id")
                for col in (pykeen.typing.LABEL_HEAD, pykeen.typing.LABEL_RELATION, pykeen.typing.LABEL_TAIL)
            )
        )
        self.df = pandas.DataFrame(data=data, columns=columns)
        self.df["score"] = torch.rand(size=(len(self.df),), generator=torch.manual_seed(seed=42)).numpy()
        return kwargs


class PredictionPostProcessorMetaTestCase(
    unittest_templates.MetaTestCase[pykeen.models.predict.PredictionPostProcessor]
):
    """Test for tests for prediction post processing."""

    base_cls = pykeen.models.predict.PredictionPostProcessor
    base_test = cases.PredictionPostProcessorTestCase


class CountScoreConsumerTestCase(cases.ScoreConsumerTests):
    """Test count score consumer."""

    cls = pykeen.models.predict.CountScoreConsumer


class TopKScoreConsumerTestCase(cases.ScoreConsumerTests):
    """Test top-k score consumer."""

    cls = pykeen.models.predict.TopKScoreConsumer


class AllScoreConsumerTestCase(cases.ScoreConsumerTests):
    """Test all score consumer."""

    cls = pykeen.models.predict.AllScoreConsumer
    kwargs = dict(
        num_entities=cases.ScoreConsumerTests.num_entities,
        num_relations=cases.ScoreConsumerTests.num_entities,
    )


class ScoreConsumerMetaTestCase(unittest_templates.MetaTestCase[pykeen.models.predict.ScoreConsumer]):
    """Test for tests for score consumers."""

    base_cls = pykeen.models.predict.ScoreConsumer
    base_test = cases.ScoreConsumerTests


@pytest.mark.parametrize(["num_entities", "num_relations"], [(3, 2)])
def test_consume_scores(num_entities: int, num_relations: int):
    """Test for consume_scores."""
    dataset = pykeen.models.predict.AllPredictionDataset(num_entities=num_entities, num_relations=num_relations)
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=False)
    )
    consumer = pykeen.models.predict.CountScoreConsumer()
    pykeen.models.predict.consume_scores(model, dataset, consumer)
    assert consumer.batch_count == num_relations * num_entities
    assert consumer.score_count == num_relations * num_entities**2


@pytest.mark.parametrize(
    ["k", "target"],
    itertools.product([None, 2], [pykeen.typing.LABEL_HEAD, pykeen.typing.LABEL_RELATION, pykeen.typing.LABEL_TAIL]),
)
def test_predict(k: Optional[int], target: pykeen.typing.Target):
    """Test the predict method."""
    num_entities, num_relations = 3, 2
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=False)
    )
    pykeen.models.predict.predict(model=model, k=k, target=target)
