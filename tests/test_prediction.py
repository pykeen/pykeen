"""Tests for prediction tools."""
import functools
import itertools
from typing import Any, MutableMapping, Optional, Tuple

import pandas
import pytest
import torch
import unittest_templates

import pykeen.models.mocks
import pykeen.predict
import pykeen.typing
from pykeen.triples.triples_factory import KGInfo
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


@pytest.mark.parametrize(
    ["k", "target", "batch_size"],
    itertools.product(
        [None, 2], [pykeen.typing.LABEL_HEAD, pykeen.typing.LABEL_RELATION, pykeen.typing.LABEL_TAIL], [1, 2]
    ),
)
def test_predict(k: Optional[int], target: pykeen.typing.Target, batch_size: int):
    """Test the predict method."""
    num_entities, num_relations = 3, 2
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=False)
    )
    pykeen.predict.predict_all(model=model, k=k, target=target, batch_size=batch_size)


class InverseRelationPredictionTests(unittest_templates.GenericTestCase[pykeen.models.FixedModel]):
    """Test for prediction with inverse relations."""

    cls = pykeen.models.FixedModel

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        # create triples factory with inverse relations
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = self.factory = Nations(create_inverse_triples=True).training
        return kwargs

    def _combination_batch(
        self,
        heads: bool = True,
        relations: bool = True,
        tails: bool = True,
    ) -> torch.LongTensor:
        """Generate a batch with all combinations."""
        factors = []
        if heads:
            factors.append(range(self.factory.num_entities))
        if relations:
            factors.append(range(self.factory.real_num_relations))
        if tails:
            factors.append(range(self.factory.num_entities))
        return torch.as_tensor(
            data=list(itertools.product(*factors)),
            dtype=torch.long,
        )

    def test_predict_hrt(self):
        """Test predict_hrt."""
        hrt_batch = self._combination_batch()
        expected_scores = self.instance._generate_fake_scores(
            h=hrt_batch[:, 0],
            r=2 * hrt_batch[:, 1],
            t=hrt_batch[:, 2],
        ).unsqueeze(dim=-1)
        scores = self.instance.predict_hrt(hrt_batch=hrt_batch)
        assert torch.allclose(scores, expected_scores)

    def test_predict_h(self):
        """Test predict_h."""
        rt_batch = self._combination_batch(heads=False)
        # head prediction via inverse tail prediction
        expected_scores = self.instance._generate_fake_scores(
            h=rt_batch[:, 1, None],
            r=2 * rt_batch[:, 0, None] + 1,
            t=torch.arange(self.factory.num_entities).unsqueeze(dim=0),
        )
        scores = self.instance.predict_h(rt_batch=rt_batch)
        assert torch.allclose(scores, expected_scores)

    def test_predict_t(self):
        """Test predict_t."""
        hr_batch = self._combination_batch(tails=False)
        expected_scores = self.instance._generate_fake_scores(
            h=hr_batch[:, 0, None],
            r=2 * hr_batch[:, 1, None],
            t=torch.arange(self.factory.num_entities).unsqueeze(dim=0),
        )
        scores = self.instance.predict_t(hr_batch=hr_batch)
        assert torch.allclose(scores, expected_scores)


# TODO: update
class MovedTest:
    def _test_score_all_triples(self, k: Optional[int], batch_size: int = 16):
        """Test score_all_triples.

        :param k: The number of triples to return. Set to None, to keep all.
        :param batch_size: The batch size to use for calculating scores.
        """
        top_triples, top_scores = predict_all(model=self.instance, batch_size=batch_size, k=k)

        # check type
        assert torch.is_tensor(top_triples)
        assert torch.is_tensor(top_scores)
        assert top_triples.dtype == torch.long
        assert top_scores.dtype == torch.float32

        # check shape
        actual_k, n_cols = top_triples.shape
        assert n_cols == 3
        if k is None:
            assert actual_k == self.factory.num_entities**2 * self.factory.num_relations
        else:
            assert actual_k == min(k, self.factory.num_triples)
        assert top_scores.shape == (actual_k,)

        # check ID ranges
        assert (top_triples >= 0).all()
        assert top_triples[:, [0, 2]].max() < self.instance.num_entities
        assert top_triples[:, 1].max() < self.instance.num_relations

    def test_score_all_triples(self):
        """Test score_all_triples with a large batch size."""
        # this is only done in one of the models
        self._test_score_all_triples(k=15, batch_size=16)

    def test_score_all_triples_singleton_batch(self):
        """Test score_all_triples with a batch size of 1."""
        self._test_score_all_triples(k=15, batch_size=1)

    def test_score_all_triples_large_batch(self):
        """Test score_all_triples with a batch size larger than k."""
        self._test_score_all_triples(k=10, batch_size=16)

    def test_score_all_triples_keep_all(self):
        """Test score_all_triples with k=None."""
        # this is only done in one of the models
        self._test_score_all_triples(k=None)

    def test_predict(self):
        """Test prediction workflow with inverse relations."""
        predict_all(model=self.instance, k=10)

    def test_get_all_prediction_df(self):
        """Test consistency of top-k scoring."""
        ks = [5, 10]
        dfs = [
            get_all_prediction(
                model=self.instance,
                triples_factory=self.factory,
                batch_size=1,
                k=k,
            )
            .nlargest(n=min(ks), columns="score")
            .reset_index(drop=True)
            for k in ks
        ]
        assert set(dfs[0].columns) == set(dfs[0].columns)
        for column in dfs[0].columns:
            numpy.testing.assert_equal(dfs[0][column].values, dfs[1][column].values)
