# -*- coding: utf-8 -*-

"""Test the evaluators."""

import itertools
import unittest
from operator import itemgetter
from typing import Any, Collection, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy
import numpy.random
import numpy.testing
import pandas
import pytest
import torch
import unittest_templates
from more_itertools import pairwise

from pykeen.constants import COLUMN_LABELS
from pykeen.datasets import Nations
from pykeen.evaluation import Evaluator, MetricResults, OGBEvaluator, RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.classification_evaluator import (
    CLASSIFICATION_METRICS,
    ClassificationEvaluator,
    ClassificationMetricResults,
)
from pykeen.evaluation.evaluator import (
    create_dense_positive_mask_,
    create_sparse_positive_filter_,
    filter_scores_,
    get_candidate_set_size,
    prepare_filter_triples,
)
from pykeen.evaluation.rank_based_evaluator import (
    MacroRankBasedEvaluator,
    SampledRankBasedEvaluator,
    sample_negatives,
    summarize_values,
)
from pykeen.evaluation.ranking_metric_lookup import MetricKey
from pykeen.evaluation.ranks import Ranks
from pykeen.metrics.ranking import (
    AdjustedArithmeticMeanRankIndex,
    ArithmeticMeanRank,
    HitsAtK,
    InverseHarmonicMeanRank,
    rank_based_metric_resolver,
)
from pykeen.models import FixedModel
from pykeen.typing import (
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    RANK_OPTIMISTIC,
    RANK_PESSIMISTIC,
    RANK_REALISTIC,
    RANK_TYPES,
    SIDE_BOTH,
    SIDES,
    MappedTriples,
    Target,
)
from tests import cases, mocks
from tests.utils import needs_packages


@pytest.mark.parametrize(["estimator", "ci"], [(numpy.mean, 60), ("mean", "std"), (numpy.mean, numpy.var)])
def test_summarize_values(estimator, ci):
    """Test value summarization."""
    gen = numpy.random.default_rng(seed=42)
    vs = gen.random(size=(17,)).tolist()
    r = summarize_values(vs=vs, estimator=estimator, ci=ci)
    assert isinstance(r, tuple)
    assert len(r) == 2
    center, variation = r
    assert isinstance(center, float)
    assert isinstance(variation, float)


class RankBasedEvaluatorTests(cases.EvaluatorTestCase):
    """unittest for the RankBasedEvaluator."""

    cls = RankBasedEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, RankBasedMetricResults)
        # check correct num_entities
        assert self.instance.num_entities == self.dataset.num_entities
        result: RankBasedMetricResults

        for (metric, side, rank_type), value in result.data.items():
            self.assertIn(side, SIDES)
            self.assertIn(rank_type, RANK_TYPES)
            self.assertIsInstance(metric, str)
            self.assertIsInstance(value, (float, int))

    def test_finalize_multi(self) -> None:
        """Test multi finalize."""
        self._process_batches()
        assert isinstance(self.instance, RankBasedEvaluator)
        n_boot = 3
        result = self.instance.finalize_multi(n_boot=n_boot)
        # check type
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, list) for v in result.values())
        # check length
        assert all(len(v) == n_boot for v in result.values())

    def test_finalize_with_confidence(self):
        """Test finalization with confidence estimation."""
        self._process_batches()
        assert isinstance(self.instance, RankBasedEvaluator)
        result = self.instance.finalize_with_confidence(n_boot=3)
        # check type
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, tuple) for v in result.values())
        # check length
        assert all(len(v) == 2 for v in result.values())
        # check confidence positivity
        assert all(c >= 0 for _, c in result.values())


class SampledRankBasedEvaluatorTests(RankBasedEvaluatorTests):
    """unittest for the SampledRankBasedEvaluator."""

    cls = SampledRankBasedEvaluator
    kwargs = dict(num_negatives=3)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["evaluation_factory"] = self.factory
        kwargs["additional_filter_triples"] = self.dataset.training.mapped_triples
        return kwargs


@needs_packages("ogb")
class OGBEvaluatorTests(RankBasedEvaluatorTests):
    """Unit test for OGB evaluator."""

    cls = OGBEvaluator
    kwargs = dict(num_negatives=3)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["evaluation_factory"] = self.factory
        kwargs["batch_size"] = 1
        return kwargs

    def test_ogb_evaluate_alternate(self):
        """Test OGB evaluation."""
        self.instance: SampledRankBasedEvaluator
        model = FixedModel(triples_factory=self.factory)
        result = self.instance.evaluate(model=model, mapped_triples=self.factory.mapped_triples, batch_size=1)
        assert isinstance(result, MetricResults)


class MacroRankBasedEvaluatorTests(RankBasedEvaluatorTests):
    """unittest for the MacroRankBasedEvaluator."""

    cls = MacroRankBasedEvaluator


class ClassificationEvaluatorTest(cases.EvaluatorTestCase):
    """Unittest for the ClassificationEvaluator."""

    cls = ClassificationEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, ClassificationMetricResults)

        # check value
        scores = data["scores"].detach().cpu().numpy()
        mask = data["mask"].detach().cpu().float().numpy()
        batch = data["batch"].detach().cpu().numpy()

        # filtering
        mask_filtered, scores_filtered = [], []
        for group_indices in [(0, 1), (1, 2)]:
            uniq = dict()
            for i, key in enumerate(batch[:, group_indices].tolist()):
                uniq[tuple(key)] = i
            indices = sorted(uniq.values())
            mask_filtered.append(mask[indices])
            scores_filtered.append(scores[indices])
        mask = numpy.concatenate(mask_filtered, axis=0)
        scores = numpy.concatenate(scores_filtered, axis=0)

        y_true, y_score = numpy.array(mask.flat), numpy.array(scores.flat)
        for name, metric in CLASSIFICATION_METRICS.items():
            with self.subTest(metric=name):
                exp_score = metric.score(y_true=y_true, y_score=y_score)
                self.assertIn(name, result.data)
                act_score = result.get_metric(name)
                if numpy.isnan(exp_score):
                    self.assertTrue(numpy.isnan(act_score))
                else:
                    self.assertAlmostEqual(act_score, exp_score, msg=f"failed for {name}", delta=7)


class EvaluatorUtilsTests(unittest.TestCase):
    """Test the utility functions used by evaluators."""

    def setUp(self) -> None:
        """Set up the test case with a fixed random seed."""
        self.generator = torch.random.manual_seed(seed=42)

    def test_compute_rank_from_scores(self):
        """Test the _compute_rank_from_scores() function."""
        batch_size = 3
        all_scores = torch.tensor(
            [
                [2.0, 2.0, 1.0, 3.0, 5.0],
                [1.0, 1.0, 3.0, 4.0, 0.0],
                [1.0, 1.0, 3.0, float("nan"), 0],
            ]
        )
        # true_score: (2, 3, 3)
        true_score = torch.as_tensor([2.0, 3.0, 3.0]).view(batch_size, 1)
        exp_best_rank = torch.as_tensor([3.0, 2.0, 1.0])
        exp_worst_rank = torch.as_tensor([4.0, 2.0, 1.0])
        exp_avg_rank = 0.5 * (exp_best_rank + exp_worst_rank)
        exp_number_of_options = torch.as_tensor([5, 5, 4])
        ranks = Ranks.from_scores(true_score=true_score, all_scores=all_scores)

        optimistic_rank = ranks.optimistic
        assert optimistic_rank.shape == (batch_size,)
        assert (optimistic_rank == exp_best_rank).all()

        pessimistic_rank = ranks.pessimistic
        assert pessimistic_rank.shape == (batch_size,)
        assert (pessimistic_rank == exp_worst_rank).all()

        realistic_rank = ranks.realistic
        assert realistic_rank.shape == (batch_size,)
        assert (realistic_rank == exp_avg_rank).all(), (realistic_rank, exp_avg_rank)

        number_of_options = ranks.number_of_options
        assert number_of_options is not None
        assert number_of_options.shape == (batch_size,)
        assert (number_of_options == exp_number_of_options).all(), (number_of_options, exp_number_of_options)

    def test_create_sparse_positive_filter_(self):
        """Test method create_sparse_positive_filter_."""
        batch_size = 4
        factory = Nations().training
        all_triples = factory.mapped_triples
        batch = all_triples[:batch_size, :]

        # head based filter
        sparse_positives, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_triples,
            relation_filter=None,
            filter_col=0,
        )

        # preprocessing for faster lookup
        triples = set()
        for trip in all_triples.detach().numpy():
            triples.add(tuple(map(int, trip)))

        # check that all found positives are positive
        for batch_id, entity_id in sparse_positives:
            same = batch[batch_id, 1:]
            assert (int(entity_id),) + tuple(map(int, same)) in triples

    def test_create_dense_positive_mask_(self):
        """Test method create_dense_positive_mask_."""
        batch_size = 3
        num_positives = 5
        num_entities = 7
        zero_tensor = torch.zeros(batch_size, num_entities)
        filter_batch = torch.empty(num_positives, 2, dtype=torch.long)
        for i in range(num_positives):
            filter_batch[i, 0] = i % batch_size
        filter_batch[:, 1] = torch.randperm(num_positives, generator=self.generator)
        dense_mask = create_dense_positive_mask_(zero_tensor=zero_tensor, filter_batch=filter_batch)

        # check in-place
        assert id(dense_mask) == id(zero_tensor)

        for b in range(batch_size):
            for e in range(num_entities):
                if (torch.as_tensor([b, e]).view(1, 2) == filter_batch).all(dim=1).any():
                    assert dense_mask[b, e] == 1
                else:
                    assert dense_mask[b, e] == 0

    def test_filter_corrupted_triples(self):
        """Test the filter_corrupted_triples() function."""
        batch_size = 2
        num_entities = 4
        all_pos_triples = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [1, 3, 3],
                [3, 4, 1],
                [0, 2, 2],
                [3, 1, 2],
                [1, 2, 0],
            ],
            dtype=torch.long,
        )
        batch = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        )
        head_filter_mask = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
            ],
            dtype=torch.bool,
        )
        tail_filter_mask = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, True],
            ],
            dtype=torch.bool,
        )
        exp_head_filter_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
            ],
            dtype=torch.bool,
        )
        exp_tail_filter_mask = torch.tensor(
            [
                [False, False, True, False],
                [True, False, False, True],
            ],
            dtype=torch.bool,
        )
        assert batch.shape == (batch_size, 3)
        assert head_filter_mask.shape == (batch_size, num_entities)
        assert tail_filter_mask.shape == (batch_size, num_entities)

        # Test head scores
        head_scores = torch.randn(batch_size, num_entities, generator=self.generator)
        old_head_scores = head_scores.detach().clone()
        positive_filter_heads, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=None,
            filter_col=0,
        )
        filtered_head_scores = filter_scores_(
            scores=head_scores,
            filter_batch=positive_filter_heads,
        )
        # Assert in-place modification
        mask = torch.isfinite(head_scores)
        assert (head_scores[mask] == filtered_head_scores[mask]).all()
        assert not torch.isfinite(filtered_head_scores[~mask]).any()

        # Assert correct filtering
        assert (old_head_scores[~exp_head_filter_mask] == filtered_head_scores[~exp_head_filter_mask]).all()
        assert not torch.isfinite(filtered_head_scores[exp_head_filter_mask]).any()

        # Test tail scores
        tail_scores = torch.randn(batch_size, num_entities, generator=self.generator)
        old_tail_scores = tail_scores.detach().clone()
        positive_filter_tails, _ = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=2,
        )
        filtered_tail_scores = filter_scores_(
            scores=tail_scores,
            filter_batch=positive_filter_tails,
        )
        # Assert in-place modification
        mask = torch.isfinite(tail_scores)
        assert (tail_scores[mask] == filtered_tail_scores[mask]).all()
        assert not torch.isfinite(filtered_tail_scores[~mask]).any()

        # Assert correct filtering
        assert (old_tail_scores[~exp_tail_filter_mask] == filtered_tail_scores[~exp_tail_filter_mask]).all()
        assert not torch.isfinite(filtered_tail_scores[exp_tail_filter_mask]).any()


class DummyEvaluator(Evaluator):
    """A dummy evaluator for testing the structure of the evaluation function."""

    def __init__(self, *, counter: int, filtered: bool, automatic_memory_optimization: bool = True) -> None:
        super().__init__(filtered=filtered, automatic_memory_optimization=automatic_memory_optimization)
        self.counter = counter

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if target == LABEL_TAIL:
            self.counter += 1
        elif target == LABEL_HEAD:
            self.counter -= 1

    def clear(self):  # noqa: D102
        pass

    def finalize(self) -> MetricResults:  # noqa: D102
        return RankBasedMetricResults(
            dict(
                arithmetic_mean_rank=self.counter,
                geometric_mean_rank=None,
                harmonic_mean_rank=None,
                median_rank=None,
                inverse_arithmetic_mean_rank=None,
                inverse_geometric_mean_rank=None,
                inverse_harmonic_mean_rank=None,
                inverse_median_rank=None,
                rank_std=None,
                rank_var=None,
                rank_mad=None,
                rank_count=None,
                adjusted_arithmetic_mean_rank=None,
                adjusted_arithmetic_mean_rank_index=None,
                hits_at_k=dict(),
            )
        )

    def __repr__(self):  # noqa: D105
        return f"{self.__class__.__name__}(losses={self.losses})"


class TestEvaluationStructure(unittest.TestCase):
    """Tests for testing the correct structure of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation structure."""
        self.counter = 1337
        self.evaluator = DummyEvaluator(counter=self.counter, filtered=True, automatic_memory_optimization=False)
        self.dataset = Nations()
        self.model = FixedModel(triples_factory=self.dataset.training)

    def test_evaluation_structure(self):
        """Test if the evaluator has a balanced call of head and tail processors."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            additional_filter_triples=self.dataset.training.mapped_triples,
            mapped_triples=self.dataset.testing.mapped_triples,
            batch_size=1,
            use_tqdm=False,
        )
        self.assertIsInstance(eval_results, RankBasedMetricResults)
        assert eval_results.arithmetic_mean_rank == self.counter, "Should end at the same value as it started"


class TestEvaluationFiltering(unittest.TestCase):
    """Tests for testing the correct filtering of positive triples of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation filtering."""
        self.evaluator = RankBasedEvaluator(filtered=True, automatic_memory_optimization=False)
        self.triples_factory = Nations().training
        self.model = FixedModel(triples_factory=self.triples_factory)

        # The MockModel gives the highest score to the highest entity id
        max_score = self.triples_factory.num_entities - 1

        # The test triples are created to yield the third highest score on both head and tail prediction
        self.test_triples = torch.tensor([[max_score - 2, 0, max_score - 2]])

        # Write new mapped triples to the model, since the model's triples will be used to filter
        # These triples are created to yield the highest score on both head and tail prediction for the
        # test triple at hand
        self.training_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score],
                [max_score, 0, max_score - 2],
            ],
        )

        # The validation triples are created to yield the second highest score on both head and tail prediction for the
        # test triple at hand
        self.validation_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score - 1],
                [max_score - 1, 0, max_score - 2],
            ],
        )

    def test_evaluation_filtering_without_validation_triples(self):
        """Test if the evaluator's triple filtering works as expected."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.test_triples,
            additional_filter_triples=self.training_triples,
            batch_size=1,
            use_tqdm=False,
        )
        assert eval_results.get_metric(name="mr") == 2, "The mean rank should equal 2"

    def test_evaluation_filtering_with_validation_triples(self):
        """Test if the evaluator's triple filtering works as expected when including additional filter triples."""
        eval_results = self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.test_triples,
            additional_filter_triples=[
                self.training_triples,
                self.validation_triples,
            ],
            batch_size=1,
            use_tqdm=False,
        )
        assert eval_results.get_metric(name="mr") == 1, "The rank should equal 1"


def test_resolve_metric_name():
    """Test metric name resolution."""
    for s, (cls, side, rank_type, *args) in (
        (
            "mrr",
            (InverseHarmonicMeanRank, SIDE_BOTH, RANK_REALISTIC),
        ),
        (
            "both.mean_rank",
            (ArithmeticMeanRank, SIDE_BOTH, RANK_REALISTIC),
        ),
        (
            "avg.mean_rank",
            (ArithmeticMeanRank, SIDE_BOTH, RANK_REALISTIC),
        ),
        (
            "tail.worst.mean_rank",
            (ArithmeticMeanRank, LABEL_TAIL, RANK_PESSIMISTIC),
        ),
        (
            "avg.amri",
            (AdjustedArithmeticMeanRankIndex, SIDE_BOTH, RANK_REALISTIC),
        ),
        (
            "hits_at_k",
            (HitsAtK, SIDE_BOTH, RANK_REALISTIC, 10),
        ),
        (
            "head.best.hits_at_k.3",
            (HitsAtK, LABEL_HEAD, RANK_OPTIMISTIC, 3),
        ),
        (
            "hits_at_1",
            (HitsAtK, SIDE_BOTH, RANK_REALISTIC, 1),
        ),
        (
            "H@10",
            (HitsAtK, SIDE_BOTH, RANK_REALISTIC, 10),
        ),
    ):
        expected = str(MetricKey(metric=cls(*args).key, side=side, rank_type=rank_type))
        result = MetricKey.normalize(s)
        assert result == expected, s


def test_sample_negatives():
    """Test for sample_negatives."""
    dataset = Nations()
    num_negatives = 2
    evaluation_triples = dataset.validation.mapped_triples
    additional_filter_triples = dataset.training.mapped_triples
    negatives = sample_negatives(
        evaluation_triples=evaluation_triples,
        additional_filter_triples=additional_filter_triples,
        num_entities=dataset.num_entities,
        num_samples=num_negatives,
    )
    head_negatives, tail_negatives = negatives[LABEL_HEAD], negatives[LABEL_TAIL]
    num_triples = evaluation_triples.shape[0]
    true = set(
        map(
            tuple,
            prepare_filter_triples(
                mapped_triples=evaluation_triples,
                additional_filter_triples=additional_filter_triples,
            ).tolist(),
        )
    )
    for i, negatives in zip((0, 2), (head_negatives, tail_negatives)):
        assert torch.is_tensor(negatives)
        assert negatives.dtype == torch.long
        assert negatives.shape == (num_triples, num_negatives)
        # check true negatives
        full_negatives = torch.empty(num_triples, num_negatives, 3)
        full_negatives[:, :, :] = evaluation_triples[:, None, :]
        full_negatives[:, :, i] = negatives
        full_negatives = full_negatives.view(-1, 3)
        negative_set = set(map(tuple, full_negatives.tolist()))
        assert negative_set.isdisjoint(true)
        # TODO: check no repetitions (if possible)


class CandidateSetSizeTests(unittest.TestCase):
    """Tests for candidate set size calculation."""

    def setUp(self) -> None:
        """Prepare the test data."""
        self.dataset = Nations()

    def _test_get_candidate_set_size(
        self,
        mapped_triples: MappedTriples,
        restrict_entities_to: Optional[Collection[int]],
        restrict_relations_to: Optional[Collection[int]],
        additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]],
        num_entities: Optional[int],
    ):
        """Test get_candidate_set_size."""
        df = get_candidate_set_size(
            mapped_triples=mapped_triples,
            restrict_entities_to=restrict_entities_to,
            restrict_relations_to=restrict_relations_to,
            additional_filter_triples=additional_filter_triples,
            num_entities=num_entities,
        )
        # return type
        assert isinstance(df, pandas.DataFrame)
        # columns
        assert set(df.columns) == {
            "index",
            LABEL_HEAD,
            LABEL_RELATION,
            LABEL_TAIL,
            f"{LABEL_HEAD}_candidates",
            f"{LABEL_TAIL}_candidates",
        }
        # value range
        if not restrict_entities_to and not restrict_relations_to:
            numpy.testing.assert_array_equal(df["index"], numpy.arange(mapped_triples.shape[0]))
            numpy.testing.assert_array_equal(df[list(COLUMN_LABELS)].values, mapped_triples.numpy())
        for candidate_column in (f"{LABEL_HEAD}_candidates", f"{LABEL_TAIL}_candidates"):
            numpy.testing.assert_array_less(-1, df[candidate_column])
            numpy.testing.assert_array_less(df[candidate_column], self.dataset.num_entities)

    def test_simple(self):
        """Test the simple case: nothing to restrict or filter or infer."""
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            None,
            None,
            None,
            self.dataset.num_entities,
        )

    def test_entity_restriction(self):
        """Test with entity restriction."""
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            {0, 1},
            None,
            None,
            self.dataset.num_entities,
        )

    def test_relation_restriction(self):
        """Test with relation restriction."""
        self._test_get_candidate_set_size(
            # relation restriction
            self.dataset.training.mapped_triples,
            None,
            {0, 1},
            None,
            self.dataset.num_entities,
        )

    def test_single_filter(self):
        """Test with additional filter triples."""
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            None,
            None,
            self.dataset.validation.mapped_triples,
            self.dataset.num_entities,
        )

    def test_multi_filter(self):
        """Test with multiple additional filter triples."""
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            None,
            None,
            (self.dataset.validation.mapped_triples, self.dataset.testing.mapped_triples),
            self.dataset.num_entities,
        )

    def test_all(self):
        """Test with filtering restriction and entity count inference."""
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            {0, 1, 2},
            {1, 2, 3},
            (self.dataset.validation.mapped_triples, self.dataset.testing.mapped_triples),
            None,
        )

    def test_entity_count_inference(self):
        """Test inference of entity count."""
        # with explicit num_entities
        df = get_candidate_set_size(
            mapped_triples=self.dataset.training.mapped_triples,
            num_entities=self.dataset.num_entities,
        )
        # with inferred num_entities
        df2 = get_candidate_set_size(
            mapped_triples=self.dataset.training.mapped_triples,
            num_entities=None,
        )
        for column in df.columns:
            numpy.testing.assert_array_equal(df[column], df2[column])


class ExpectedMetricsTests(unittest.TestCase):
    """Tests for expected metrics."""

    def _iter_num_candidates(self) -> Iterable[Tuple[Tuple[int, ...], int]]:
        """Generate number of ranking candidate arrays of different shapes."""
        generator: numpy.random.Generator = numpy.random.default_rng(seed=42)
        # test different shapes
        for shape, total in (
            (tuple(), 20),
            ((10, 2), 275),
            ((10_000,), 1237),
        ):
            yield generator.integers(low=1, high=total, size=shape), total

    def test_expected_mean_rank(self):
        """Test expected_mean_rank."""
        metric = ArithmeticMeanRank()
        # test different shapes
        for num_candidates, total in self._iter_num_candidates():
            emr = metric.expected_value(num_candidates=num_candidates)
            # value range
            assert emr >= 0
            assert emr <= total

    def test_expected_hits_at_k(self):
        """Test expected Hits@k."""
        for k, (num_candidates, total) in itertools.product(
            (1, 3, 100),
            self._iter_num_candidates(),
        ):
            metric = HitsAtK(k=k)
            ehk = metric.expected_value(num_candidates=num_candidates)
            # value range
            assert ehk >= 0
            assert ehk <= 1.0
            if total <= k:
                self.assertAlmostEqual(ehk, 1.0)

    def test_expected_hits_at_k_manual(self):
        """Test expected Hits@k, where some candidate set sizes are smaller than k, but not all."""
        metric = HitsAtK(k=10)
        self.assertAlmostEqual(metric.expected_value(num_candidates=[5, 20]), (1 + 0.5) / 2)


def test_prepare_filter_triples():
    """Tests for prepare_filter_triples."""
    dataset = Nations()
    mapped_triples = dataset.testing.mapped_triples
    for additional_filter_triples in (
        None,  # no additional
        dataset.validation.mapped_triples,  # single tensor
        [dataset.validation.mapped_triples, dataset.training.mapped_triples],  # multiple tensors
    ):
        filter_triples = prepare_filter_triples(
            mapped_triples=mapped_triples,
            additional_filter_triples=additional_filter_triples,
        )
        assert torch.is_tensor(filter_triples)
        assert filter_triples.ndim == 2
        assert filter_triples.shape[1] == 3
        assert filter_triples.shape[0] >= mapped_triples.shape[0]
        # check unique
        assert filter_triples.unique(dim=0).shape == filter_triples.shape


class RankBasedMetricResultTests(cases.MetricResultTestCase):
    """Tests for rank-based metric results."""

    cls = RankBasedMetricResults
    num_entities: int = 7
    num_triples: int = 13

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["data"] = RankBasedMetricResults.create_random().data
        return kwargs

    def _verify_flat_dict(self, flat_dict: Mapping[str, Any]):  # noqa: D102
        for metric_cls in rank_based_metric_resolver:
            metric = metric_cls()
            metric_name = metric.key
            self.assertTrue(any(metric_name in key for key in flat_dict.keys()), metric_name)

    def test_monotonicity_in_rank_type(self):
        """Test monotonicity for different rank-types."""
        hits_prefixes = [
            "hits_at_",
            "z_hits_at_",
            "adjusted_hits_at_",
        ]
        self.instance: RankBasedMetricResults
        metric_names, targets = [set(map(itemgetter(i), self.instance.data.keys())) for i in (0, 1)]
        for metric_name in metric_names:
            if metric_name in {"variance", "standard_deviation", "median_absolute_deviation"}:
                continue
            norm_metric_name = metric_name
            for hits_prefix in hits_prefixes:
                if metric_name.startswith(hits_prefix):
                    # strips off the "k" at the end
                    norm_metric_name = hits_prefix
            increasing = rank_based_metric_resolver.lookup(norm_metric_name).increasing
            exp_sort_indices = [0, 1, 2] if increasing else [2, 1, 0]
            for target in targets:
                values = numpy.asarray(
                    [
                        self.instance.data[metric_name, target, rank_type]
                        for rank_type in (RANK_PESSIMISTIC, RANK_REALISTIC, RANK_OPTIMISTIC)
                    ]
                )
                for i, j in pairwise(exp_sort_indices):
                    assert values[i] <= values[j], metric_name

    def test_to_df(self):
        """Test to_df."""
        df = self.instance.to_df()
        assert isinstance(df, pandas.DataFrame)


class ClassificationMetricResultsTests(cases.MetricResultTestCase):
    """Tests for classification metric results."""

    cls = ClassificationMetricResults
    num_entities: int = 7
    num_triples: int = 13

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        # Populate with real results.
        evaluator = ClassificationEvaluator()
        evaluator.process_scores_(
            hrt_batch=torch.randint(self.num_entities, size=(self.num_triples, 3)),
            target=LABEL_TAIL,
            scores=torch.rand(self.num_triples, self.num_entities),
            dense_positive_mask=torch.rand(self.num_triples, self.num_entities) < 0.5,
        )
        kwargs["data"] = evaluator.finalize().data
        return kwargs


class MetricResultMetaTestCase(unittest_templates.MetaTestCase):
    """Test for tests for metric results."""

    base_cls = MetricResults
    base_test = cases.MetricResultTestCase


class EvaluatorMetaTestCase(unittest_templates.MetaTestCase):
    """Test for tests for evaluators."""

    base_cls = Evaluator
    base_test = cases.EvaluatorTestCase
    skip_cls = {
        mocks.MockEvaluator,
        DummyEvaluator,
    }
