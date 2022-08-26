"""Tests for prediction tools."""
import itertools
from typing import Optional, Tuple

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


# prediction post-processing
class TargetPredictionsTests(cases.PredictionTestCase):
    """Tests for target prediction post-processing."""

    cls = pykeen.predict.TargetPredictions
    kwargs = dict(target=pykeen.typing.LABEL_HEAD, other_columns_fixed_ids=(0, 1))


class TriplePredictionsTest(cases.PredictionTestCase):
    """Tests for triple scoring post-processing."""

    cls = pykeen.predict.TriplePredictions

    # def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    #     kwargs = super()._pre_instantiation_hook(kwargs)
    #     # mock prediction data frame
    #     data = list(
    #         map(
    #             functools.partial(sum, start=tuple()),
    #             itertools.product(
    #                 self.dataset.entity_to_id.items(),
    #                 self.dataset.relation_to_id.items(),
    #                 self.dataset.entity_to_id.items(),
    #             ),
    #         )
    #     )
    #     columns = list(
    #         itertools.chain.from_iterable(
    #             (f"{col}_label", f"{col}_id")
    #             for col in (pykeen.typing.LABEL_HEAD, pykeen.typing.LABEL_RELATION, pykeen.typing.LABEL_TAIL)
    #         )
    #     )
    #     self.df = pandas.DataFrame(data=data, columns=columns)
    #     self.df["score"] = torch.rand(size=(len(self.df),), generator=torch.manual_seed(seed=42)).numpy()
    #     return kwargs


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


@pytest.mark.parametrize(
    ["k", "target", "batch_size"],
    itertools.product(
        [None, 2], [pykeen.typing.LABEL_HEAD, pykeen.typing.LABEL_RELATION, pykeen.typing.LABEL_TAIL], [1, 2]
    ),
)
def test_predict_all(k: Optional[int], target: pykeen.typing.Target, batch_size: int):
    """Test the predict method."""
    num_entities, num_relations = 3, 2
    model = pykeen.models.mocks.FixedModel(
        triples_factory=KGInfo(num_entities=num_entities, num_relations=num_relations, create_inverse_triples=False)
    )
    pack = pykeen.predict.predict_all(model=model, k=k, target=target, batch_size=batch_size)
    assert isinstance(pack, pykeen.predict.ScorePack)
    # check type
    assert isinstance(pack.result, torch.Tensor)
    assert pack.result.dtype == torch.long
    assert isinstance(pack.scores, torch.Tensor)
    assert pack.scores.is_floating_point()
    # check shape
    if k is None:
        n = num_entities**2 * num_relations
    else:
        n = k
    assert pack.result.shape == (n, 3)
    assert pack.scores.shape == (n,)
    # check ID ranges
    assert (pack.result >= 0).all()
    assert pack.result[:, [0, 2]].max() < num_entities
    assert pack.result[:, 1].max() < num_relations


# class InverseRelationPredictionTests(unittest_templates.GenericTestCase[pykeen.models.FixedModel]):
#     """Test for prediction with inverse relations."""

#     cls = pykeen.models.FixedModel

#     def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
#         # create triples factory with inverse relations
#         kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
#         kwargs["triples_factory"] = self.factory = Nations(create_inverse_triples=True).training
#         return kwargs

#     def _combination_batch(
#         self,
#         heads: bool = True,
#         relations: bool = True,
#         tails: bool = True,
#     ) -> torch.LongTensor:
#         """Generate a batch with all combinations."""
#         factors = []
#         if heads:
#             factors.append(range(self.factory.num_entities))
#         if relations:
#             factors.append(range(self.factory.real_num_relations))
#         if tails:
#             factors.append(range(self.factory.num_entities))
#         return torch.as_tensor(
#             data=list(itertools.product(*factors)),
#             dtype=torch.long,
#         )

#     def test_predict_hrt(self):
#         """Test predict_hrt."""
#         hrt_batch = self._combination_batch()
#         expected_scores = self.instance._generate_fake_scores(
#             h=hrt_batch[:, 0],
#             r=2 * hrt_batch[:, 1],
#             t=hrt_batch[:, 2],
#         ).unsqueeze(dim=-1)
#         scores = self.instance.predict_hrt(hrt_batch=hrt_batch)
#         assert torch.allclose(scores, expected_scores)

#     def test_predict_h(self):
#         """Test predict_h."""
#         rt_batch = self._combination_batch(heads=False)
#         # head prediction via inverse tail prediction
#         expected_scores = self.instance._generate_fake_scores(
#             h=rt_batch[:, 1, None],
#             r=2 * rt_batch[:, 0, None] + 1,
#             t=torch.arange(self.factory.num_entities).unsqueeze(dim=0),
#         )
#         scores = self.instance.predict_h(rt_batch=rt_batch)
#         assert torch.allclose(scores, expected_scores)

#     def test_predict_t(self):
#         """Test predict_t."""
#         hr_batch = self._combination_batch(tails=False)
#         expected_scores = self.instance._generate_fake_scores(
#             h=hr_batch[:, 0, None],
#             r=2 * hr_batch[:, 1, None],
#             t=torch.arange(self.factory.num_entities).unsqueeze(dim=0),
#         )
#         scores = self.instance.predict_t(hr_batch=hr_batch)
#         assert torch.allclose(scores, expected_scores)


# # TODO: update
# class MovedTest:
#     def test_get_all_prediction_df(self):
#         """Test consistency of top-k scoring."""
#         ks = [5, 10]
#         dfs = [
#             get_all_prediction(
#                 model=self.instance,
#                 triples_factory=self.factory,
#                 batch_size=1,
#                 k=k,
#             )
#             .nlargest(n=min(ks), columns="score")
#             .reset_index(drop=True)
#             for k in ks
#         ]
#         assert set(dfs[0].columns) == set(dfs[0].columns)
#         for column in dfs[0].columns:
#             numpy.testing.assert_equal(dfs[0][column].values, dfs[1][column].values)
