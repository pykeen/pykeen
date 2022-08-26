"""Tests for prediction tools."""
from typing import Any, Iterable, MutableMapping, Optional, Sequence, Tuple, Union

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


def _iter_predict_triples_inputs() -> Iterable[
    Tuple[pykeen.models.Model, AnyTriples, Optional[CoreTriplesFactory], Optional[int]]
]:
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


def _iter_get_input_batch_inputs():
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


def _iter_predict_target_inputs() -> Iterable[
    Tuple[
        pykeen.models.Model,
        Union[None, int, str],
        Union[None, int, str],
        Union[None, int, str],
        Optional[CoreTriplesFactory],
        Union[None, torch.LongTensor, Sequence[Union[int, str]]],
    ]
]:
    # create model
    dataset = Nations()
    factory = dataset.training
    model = pykeen.models.mocks.FixedModel(triples_factory=factory)
    # id-based head/relation/tail prediction, no factory, no restriction
    yield model, None, 0, 1, None, None
    yield model, 1, None, 2, None, None
    yield model, 0, 1, None, None, None
    # restriction by list of ints
    yield model, 0, 1, None, None, [0, 3, 7]
    # restriction by list of tensor
    yield model, 0, None, 7, None, torch.as_tensor([0, 3, 7], dtype=torch.long)


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
    # TODO: additional verification


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


# class TestPipeline(unittest.TestCase):
#     """Test the pipeline."""

#     @classmethod
#     def setUpClass(cls):
#         """Set up a shared result."""
#         cls.device = resolve_device("cuda")
#         cls.dataset = Nations()
#         cls.result = pipeline(
#             model="TransE",
#             dataset=cls.dataset,
#             training_kwargs=dict(num_epochs=5, use_tqdm=False),
#             evaluation_kwargs=dict(use_tqdm=False),
#             device=cls.device,
#             random_seed=42,
#         )
#         cls.model = cls.result.model
#         cls.testing_mapped_triples = cls.dataset.testing.mapped_triples.to(cls.model.device)

#     def test_predict_tails_no_novelties(self):
#         """Test scoring tails without labeling as novel w.r.t. training and testing."""
#         tails_df = get_tail_prediction_df(
#             model=self.model,
#             head_label="brazil",
#             relation_label="intergovorgs",
#             testing=self.testing_mapped_triples,
#             triples_factory=self.dataset.training,
#             add_novelties=False,
#         )
#         self.assertEqual(["tail_id", "tail_label", "score"], list(tails_df.columns))
#         self.assertEqual(len(self.dataset.training.entity_to_id), len(tails_df.index))

#     def test_predict_tails_remove_known(self):
#         """Test scoring tails while removing non-novel triples w.r.t. training and testing."""
#         tails_df = get_tail_prediction_df(
#             model=self.model,
#             head_label="brazil",
#             relation_label="intergovorgs",
#             testing=self.testing_mapped_triples,
#             remove_known=True,
#             triples_factory=self.dataset.training,
#         )
#         self.assertEqual(["tail_id", "tail_label", "score"], list(tails_df.columns))
#         self.assertEqual({"jordan", "brazil", "ussr", "burma", "china"}, set(tails_df["tail_label"]))

#     def test_predict_tails_with_novelties(self):
#         """Test scoring tails with labeling as novel w.r.t. training and testing."""
#         tails_df = get_tail_prediction_df(
#             model=self.model,
#             head_label="brazil",
#             relation_label="intergovorgs",
#             triples_factory=self.dataset.training,
#             testing=self.testing_mapped_triples,
#         )
#         self.assertEqual(["tail_id", "tail_label", "score", "in_training", "in_testing"], list(tails_df.columns))
#         self.assertEqual(self.model.num_entities, len(tails_df.index))
#         training_tails = set(tails_df.loc[tails_df["in_training"], "tail_label"])
#         self.assertEqual({"usa", "uk", "netherlands", "egypt", "india", "israel", "indonesia"}, training_tails)
#         testing_tails = set(tails_df.loc[tails_df["in_testing"], "tail_label"])
#         self.assertEqual({"poland", "cuba"}, testing_tails)

#     def test_predict_relations_with_novelties(self):
#         """Test scoring relations with labeling as novel w.r.t. training and testing."""
#         rel_df = get_relation_prediction_df(
#             model=self.model,
#             head_label="brazil",
#             tail_label="uk",
#             triples_factory=self.dataset.training,
#             testing=self.testing_mapped_triples,
#         )
#         self.assertEqual(["relation_id", "relation_label", "score", "in_training", "in_testing"], list(rel_df.columns))
#         self.assertEqual(self.model.num_relations, len(rel_df.index))
#         training_rels = set(rel_df.loc[rel_df["in_training"], "relation_label"])
#         self.assertEqual(
#             {
#                 "weightedunvote",
#                 "relexports",
#                 "intergovorgs",
#                 "timesinceally",
#                 "exports3",
#                 "booktranslations",
#                 "relbooktranslations",
#                 "reldiplomacy",
#                 "ngoorgs3",
#                 "ngo",
#                 "relngo",
#                 "reltreaties",
#                 "independence",
#                 "intergovorgs3",
#                 "unweightedunvote",
#                 "commonbloc2",
#                 "relintergovorgs",
#             },
#             training_rels,
#         )
#         testing_heads = set(rel_df.loc[rel_df["in_testing"], "relation_label"])
#         self.assertEqual({"embassy"}, testing_heads)

#     def test_predict_heads_with_novelties(self):
#         """Test scoring heads with labeling as novel w.r.t. training and testing."""
#         heads_df = get_head_prediction_df(
#             model=self.model,
#             relation_label="conferences",
#             tail_label="brazil",
#             triples_factory=self.dataset.training,
#             testing=self.testing_mapped_triples,
#         )
#         self.assertEqual(["head_id", "head_label", "score", "in_training", "in_testing"], list(heads_df.columns))
#         self.assertEqual(self.model.num_entities, len(heads_df.index))
#         training_heads = set(heads_df.loc[heads_df["in_training"], "head_label"])
#         self.assertEqual({"usa", "india", "ussr", "poland", "cuba"}, training_heads)
#         testing_heads = set(heads_df.loc[heads_df["in_testing"], "head_label"])
#         self.assertEqual(set(), testing_heads)

#     def test_predict_all_no_novelties(self):
#         """Test scoring all triples without labeling as novel w.r.t. training and testing."""
#         all_df = get_all_prediction(
#             model=self.model,
#             triples_factory=self.dataset.training,
#             testing=self.testing_mapped_triples,
#             add_novelties=False,
#         )
#         self.assertIsInstance(all_df, pd.DataFrame)
#         self.assertEqual(
#             ["head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label", "score"],
#             list(all_df.columns),
#         )
#         possible = self.dataset.training.num_relations * self.model.num_entities**2
#         self.assertEqual(possible, len(all_df.index))

#     def test_predict_all_remove_known(self):
#         """Test scoring all triples while removing non-novel triples w.r.t. training and testing."""
#         all_df = get_all_prediction(
#             model=self.model,
#             triples_factory=self.dataset.training,
#             testing=self.testing_mapped_triples,
#             remove_known=True,
#         )
#         self.assertIsInstance(all_df, pd.DataFrame)
#         self.assertEqual(
#             ["head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label", "score"],
#             list(all_df.columns),
#         )
#         possible = self.dataset.training.num_relations * self.model.num_entities**2
#         known = self.dataset.training.num_triples + self.testing_mapped_triples.shape[0]
#         self.assertNotEqual(possible, known, msg="testing and training triples cover all possible triples")
#         self.assertEqual(possible - known, len(all_df.index))

#     def test_predict_all_with_novelties(self):
#         """Test scoring all triples with labeling as novel w.r.t. training and testing."""
#         all_df = get_all_prediction(
#             model=self.model,
#             triples_factory=self.dataset.training,
#             testing=self.testing_mapped_triples,
#         )
#         self.assertIsInstance(all_df, pd.DataFrame)
#         self.assertEqual(
#             [
#                 "head_id",
#                 "head_label",
#                 "relation_id",
#                 "relation_label",
#                 "tail_id",
#                 "tail_label",
#                 "score",
#                 "in_training",
#                 "in_testing",
#             ],
#             list(all_df.columns),
#         )
#         possible = self.dataset.training.num_relations * self.model.num_entities**2
#         self.assertEqual(possible, len(all_df.index))
#         self.assertEqual(self.dataset.training.num_triples, all_df["in_training"].sum())
#         self.assertEqual(self.testing_mapped_triples.shape[0], all_df["in_testing"].sum())

#     def test_predict_triples(self):
#         """Test scoring explicitly provided triples."""
#         for triples_factory in (None, self.dataset.training):
#             df = predict_triples(
#                 model=self.model,
#                 triples=self.testing_mapped_triples,
#                 triples_factory=triples_factory,
#             )
#             assert isinstance(df, pandas.DataFrame)
#             assert df.shape[0] == self.testing_mapped_triples.shape[0]
#             assert {"head_id", "relation_id", "tail_id", "score"}.issubset(df.columns)
