"""Test the evaluators."""

from __future__ import annotations

import itertools
import unittest
import warnings
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, MutableMapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import torch
import unittest_templates
from more_itertools import pairwise

from pykeen.constants import COLUMN_LABELS
from pykeen.datasets import Nations
from pykeen.evaluation import Evaluator, MetricResults, OGBEvaluator, RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from pykeen.evaluation.evaluator import (
    _optimal_batch_size_group_id,
    create_dense_positive_mask_,
    create_sparse_positive_filter_,
    filter_scores_,
    get_candidate_set_size,
    prepare_filter_triples,
)
from pykeen.evaluation.rank_based_evaluator import (
    MacroRankBasedEvaluator,
    RankBasedMetricKey,
    SampledRankBasedEvaluator,
    sample_negatives,
    summarize_values,
)
from pykeen.evaluation.ranks import Ranks
from pykeen.metrics.ranking import (
    HITS_METRICS,
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
    ExtendedTarget,
    MappedTriples,
    Target,
    normalize_target,
)
from tests import cases, mocks
from tests.utils import needs_packages


@pytest.mark.parametrize(("estimator", "ci"), [(np.mean, 60), ("mean", "std"), (np.mean, np.var)])
def test_summarize_values(estimator, ci):
    """Test value summarization."""
    gen = np.random.default_rng(seed=42)
    vs = gen.random(size=(17,)).tolist()
    r = summarize_values(vs=vs, estimator=estimator, ci=ci)
    assert isinstance(r, tuple)
    assert len(r) == 2
    center, variation = r
    assert isinstance(center, float)
    assert isinstance(variation, float)


class RankBasedEvaluatorTests(cases.EvaluatorTestCase):
    """unittest for the RankBasedEvaluator."""

    cls: ClassVar[type[Evaluator]] = RankBasedEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, RankBasedMetricResults)
        # check correct num_entities
        assert isinstance(self.instance, RankBasedEvaluator)
        assert self.instance.num_entities == self.dataset.num_entities

        for (side, rank_type, metric), value in result.data.items():
            assert side in SIDES
            assert rank_type in RANK_TYPES
            assert isinstance(metric, str)
            assert isinstance(value, (float, int))

    def test_finalize_multi(self) -> None:
        """Test multi finalize."""
        self._process_batches()
        assert isinstance(self.instance, RankBasedEvaluator)
        n_boot = 3
        result = self.instance.finalize_multi(n_boot=n_boot)
        # check type
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result)
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
        assert all(isinstance(k, str) for k in result)
        assert all(isinstance(v, tuple) for v in result.values())
        # check length
        assert all(len(v) == 2 for v in result.values())
        # check confidence positivity
        assert all(c >= 0 for _, c in result.values())


class SampledRankBasedEvaluatorTests(RankBasedEvaluatorTests):
    """unittest for the SampledRankBasedEvaluator."""

    cls = SampledRankBasedEvaluator
    kwargs: ClassVar[Mapping[str, Any]] = {"num_negatives": 3}

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["evaluation_factory"] = self.factory
        kwargs["additional_filter_triples"] = self.dataset.training.mapped_triples
        return kwargs


@needs_packages("ogb")
class OGBEvaluatorTests(RankBasedEvaluatorTests):
    """Unit test for OGB evaluator."""

    cls = OGBEvaluator
    kwargs: ClassVar[Mapping[str, Any]] = {"num_negatives": 3}

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
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


@needs_packages("ogb")
class OGBEvaluatorValueTests(unittest.TestCase):
    """Value-level tests for the OGB evaluator with a model with known scores."""

    def setUp(self) -> None:
        """Prepare a tiny dataset; FixedModel scores increase with h, then r, then t."""
        from pykeen.triples import TriplesFactory

        self.factory = TriplesFactory.from_labeled_triples(
            np.asarray([[f"e{h}", "r", f"e{t}"] for h, t in [(0, 1), (1, 4), (2, 3)]], dtype=str),
            entity_to_id={f"e{i}": i for i in range(5)},
            relation_to_id={"r": 0},
        )
        # triples are sorted by the factory; true heads (0, 1, 2) are all outranked by the head negatives -> rank 3 each
        # true tails (1, 4, 3) outrank all tail negatives -> rank 1 each
        self.evaluator = OGBEvaluator(
            evaluation_factory=self.factory,
            head_negatives=torch.as_tensor([[3, 4], [2, 3], [3, 4]]),
            tail_negatives=torch.as_tensor([[0, 0], [0, 3], [0, 1]]),
        )
        self.model = FixedModel(triples_factory=self.factory)

    def _evaluate(self, **kwargs) -> Mapping[Any, float]:
        return self.evaluator.evaluate(
            model=self.model, mapped_triples=self.factory.mapped_triples, use_tqdm=False, **kwargs
        ).data

    def test_head_uses_true_head(self):
        """Test that the head side scores the true head, not the true tail."""
        data = self._evaluate()
        assert data[RankBasedMetricKey(side="head", rank_type="realistic", metric="inverse_harmonic_mean_rank")] == (
            pytest.approx(1 / 3)
        )
        assert data[RankBasedMetricKey(side="tail", rank_type="realistic", metric="inverse_harmonic_mean_rank")] == (
            pytest.approx(1.0)
        )
        assert data[RankBasedMetricKey(side="both", rank_type="realistic", metric="inverse_harmonic_mean_rank")] == (
            pytest.approx(2 / 3)
        )

    def test_targets_restricts_sides(self):
        """Test that only the requested targets are evaluated."""
        data = self._evaluate(targets=("tail",))
        assert {key.side for key in data} == {"tail", "both"}

    def test_only_requested_metrics(self):
        """Test that only the evaluator's supported metrics are reported."""
        evaluator = OGBEvaluator(
            evaluation_factory=self.factory,
            head_negatives=self.evaluator.negative_samples["head"],
            tail_negatives=self.evaluator.negative_samples["tail"],
            metrics=["hits_at_k"],
            metrics_kwargs={"k": 3},
            add_defaults=False,
        )
        data = evaluator.evaluate(model=self.model, mapped_triples=self.factory.mapped_triples, use_tqdm=False).data
        assert {key.metric for key in data} == {"hits_at_3"}

    def test_unsupported_restrictions(self):
        """Test that list-valued restrictions raise the intended ValueError."""
        with pytest.raises(ValueError, match="does not support"):
            self._evaluate(restrict_entities_to=[0, 1])
        with pytest.raises(ValueError, match="does not support"):
            self._evaluate(additional_filter_triples=[self.factory.mapped_triples])


class MacroRankBasedEvaluatorTests(RankBasedEvaluatorTests):
    """unittest for the MacroRankBasedEvaluator."""

    cls = MacroRankBasedEvaluator


def _process_ranks_(evaluator: RankBasedEvaluator, true_ranks: list[int], num_entities: int, keys: list[int]) -> None:
    """Feed head-prediction scores with the given true ranks; `keys` is the tail, i.e., (relation, tail) is the key."""
    n = len(true_ranks)
    # the true score is 0 and stored in the last column; exactly (rank - 1) of the other candidates are scored higher,
    # all remaining ones are scored lower, i.e., there are no ties
    scores = torch.full((n, num_entities), -1.0)
    true_scores = torch.zeros(n, 1)
    for i, rank in enumerate(true_ranks):
        scores[i, : rank - 1] = 1.0
    scores[:, -1] = true_scores[:, 0]
    hrt_batch = torch.stack(
        [torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), torch.as_tensor(keys)], dim=-1
    )
    evaluator.process_scores_(hrt_batch=hrt_batch, target=LABEL_HEAD, scores=scores, true_scores=true_scores)


_MRR_KEY = RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=InverseHarmonicMeanRank().key)


def _make_evaluators() -> dict[str, RankBasedEvaluator]:
    dataset = Nations()
    kwargs: dict[str, Any] = {"metrics": ["mrr"], "add_defaults": False}
    return {
        "rankbased": RankBasedEvaluator(**kwargs),
        "macro": MacroRankBasedEvaluator(**kwargs),
        "sampled": SampledRankBasedEvaluator(evaluation_factory=dataset.testing, num_negatives=3, **kwargs),
    }


@pytest.mark.parametrize("name", ["rankbased", "macro", "sampled"])
def test_finalize_empty(name: str):
    """Test that finalizing an evaluator without processed scores raises an informative error."""
    evaluator = _make_evaluators()[name]
    with pytest.raises(ValueError, match="no ranks to aggregate"):
        evaluator.finalize()
    with pytest.raises(ValueError, match="no ranks to aggregate"):
        evaluator.finalize_multi(n_boot=2)


@pytest.mark.parametrize("name", ["rankbased", "macro"])
def test_finalize_twice(name: str):
    """Test that finalize clears the buffers by default, i.e., a second call fails."""
    evaluator = _make_evaluators()[name]
    _process_ranks_(evaluator, true_ranks=[1, 2, 3], num_entities=5, keys=[0, 1, 2])
    evaluator.finalize()
    with pytest.raises(ValueError, match="no ranks to aggregate"):
        evaluator.finalize()


@pytest.mark.parametrize("cls", [RankBasedEvaluator, MacroRankBasedEvaluator])
def test_clear_on_finalize_false(cls: type[RankBasedEvaluator]):
    """Test that buffers are kept, and results are reproducible, if clear_on_finalize is disabled."""
    evaluator = cls(metrics=["mrr"], add_defaults=False, clear_on_finalize=False)
    _process_ranks_(evaluator, true_ranks=[1, 2, 3, 5], num_entities=5, keys=[0, 0, 1, 2])
    first = evaluator.finalize()
    assert evaluator.ranks
    second = evaluator.finalize()
    assert first.to_flat_dict() == second.to_flat_dict()


def test_macro_finalize_multi_unique_keys():
    """Test that macro and micro bootstrap coincide, if all keys are unique, i.e., all macro weights are equal."""
    ranks, keys = [1, 2, 3, 4, 5, 1, 1, 2], list(range(8))
    results = []
    for cls in (RankBasedEvaluator, MacroRankBasedEvaluator):
        evaluator = cls(metrics=["mrr"], add_defaults=False)
        _process_ranks_(evaluator, true_ranks=ranks, num_entities=5, keys=keys)
        results.append(evaluator.finalize_multi(n_boot=5))
    micro, macro = results
    assert micro.keys() == macro.keys()
    for key, values in micro.items():
        assert values == pytest.approx(macro[key])


def test_macro_finalize_multi_uses_weights():
    """Test that the macro bootstrap is centered around the macro-, not the micro-averaged value.

    40 triples share a single key and have rank 1; 10 triples have unique keys and the worst rank (10). Thus, the
    micro-averaged MRR is (40 + 10 * 0.1) / 50 = 0.82, while the macro-averaged MRR is (1 + 10 * 0.1) / 11 = 2 / 11.
    """
    num_entities = 10
    true_ranks = [1] * 40 + [num_entities] * 10
    keys = [0] * 40 + list(range(1, 11))
    evaluator = MacroRankBasedEvaluator(metrics=["mrr"], add_defaults=False, clear_on_finalize=False)
    _process_ranks_(evaluator, true_ranks=true_ranks, num_entities=num_entities, keys=keys)
    assert evaluator.finalize().get_metric(_MRR_KEY) == pytest.approx(2 / 11)
    values = evaluator.finalize_multi(n_boot=30)[f"{SIDE_BOTH}.{RANK_REALISTIC}.{_MRR_KEY.metric}"]
    assert np.median(values) == pytest.approx(2 / 11, abs=0.1)


class ClassificationEvaluatorTest(cases.EvaluatorTestCase):
    """Unittest for the ClassificationEvaluator."""

    cls = ClassificationEvaluator

    def _validate_result(
        self,
        result: MetricResults,
        data: dict[str, torch.Tensor],
    ):
        # Check for correct class
        assert isinstance(result, ClassificationMetricResults)

        for (side, metric_name), value in result.data.items():
            assert side in SIDES
            assert isinstance(metric_name, str)
            assert isinstance(value, (float, int))


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
        sparse_positives = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_triples,
            relation_filter=None,
            filter_col=0,
        )[0]

        # preprocessing for faster lookup
        triples = set()
        for trip in all_triples.detach().numpy():
            triples.add(tuple(map(int, trip)))

        # check that all found positives are positive
        for batch_id, entity_id in sparse_positives:
            same = batch[batch_id, 1:]
            assert (int(entity_id), *tuple(map(int, same))) in triples

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


class DummyMetricResults(MetricResults[tuple[ExtendedTarget]]):
    """Dummy metric results."""

    @classmethod
    def key_from_string(cls, s: str | None) -> tuple[ExtendedTarget]:
        return (normalize_target(s),)


class DummyEvaluator(Evaluator[tuple[ExtendedTarget]]):
    """A dummy evaluator for testing the structure of the evaluation function."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dummy evaluator."""
        super().__init__(*args, **kwargs)
        self.counter: Counter[Target] = Counter()

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: torch.FloatTensor | None = None,
        dense_positive_mask: torch.FloatTensor | None = None,
    ) -> None:
        self.counter.update((target,))

    def clear(self) -> None:
        self.counter.clear()

    def finalize(self) -> MetricResults[tuple[ExtendedTarget]]:
        return DummyMetricResults(data={(target,): float(count) for target, count in self.counter.items()})


class TestEvaluationStructure(unittest.TestCase):
    """Tests for testing the correct structure of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation structure."""
        self.evaluator = DummyEvaluator(filtered=True)
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
        assert isinstance(eval_results, DummyMetricResults)
        assert eval_results.get_metric(name=LABEL_HEAD) == eval_results.get_metric(name=LABEL_TAIL), (
            "Should be evaluated on the same number of batches per side"
        )


class TestEvaluationFiltering(unittest.TestCase):
    """Tests for testing the correct filtering of positive triples of the evaluation procedure."""

    def setUp(self):
        """Prepare for testing the evaluation filtering."""
        self.evaluator = RankBasedEvaluator(filtered=True)
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

    def test_evaluation_unfiltered(self):
        """Test that the raw (unfiltered) ranking protocol is supported.

        Filtering only decides whether the *other* positives are masked out; the true score - and thus a rank - is
        available either way.
        """
        eval_results = RankBasedEvaluator(filtered=False).evaluate(
            model=self.model,
            mapped_triples=self.test_triples,
            batch_size=1,
            use_tqdm=False,
        )
        # the true entity receives the third-highest score on both sides, and nothing is filtered out
        assert eval_results.get_metric(name="mr") == 3, "The raw mean rank should equal 3"


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        (None, RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=InverseHarmonicMeanRank().key)),
        ("mrr", RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=InverseHarmonicMeanRank().key)),
        (
            "both.mean_rank",
            RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=ArithmeticMeanRank().key),
        ),
        (
            "avg.mean_rank",
            RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=ArithmeticMeanRank().key),
        ),
        (
            "tail.worst.mean_rank",
            RankBasedMetricKey(side=LABEL_TAIL, rank_type=RANK_PESSIMISTIC, metric=ArithmeticMeanRank().key),
        ),
        (
            "avg.amri",
            RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=AdjustedArithmeticMeanRankIndex().key),
        ),
        ("hits_at_k", RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=HitsAtK(k=10).key)),
        (
            "head.best.hits_at_k.3",
            RankBasedMetricKey(side=LABEL_HEAD, rank_type=RANK_OPTIMISTIC, metric=HitsAtK(k=3).key),
        ),
        ("hits_at_1", RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=HitsAtK(k=1).key)),
        ("H@10", RankBasedMetricKey(side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=HitsAtK(k=10).key)),
    ],
)
def test_resolve_metric_name(string, expected):
    """Test metric name resolution."""
    assert RankBasedMetricResults.key_from_string(string) == expected


def test_default_metric_keys_unique():
    """Test that the default metrics of the rank-based evaluator have unique keys."""
    keys = [metric.key for metric in RankBasedEvaluator().metrics]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    assert not duplicates


@pytest.mark.parametrize("cls", HITS_METRICS)
@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_hits_at_k_key_roundtrip(cls, k):
    """Test that keys of (derived) Hits@k metrics contain k, and can be resolved back to the same key."""
    metric = cls(k=k)
    assert metric.key.endswith(f"hits_at_{k}")
    assert f"k={k}" in metric.extra_repr()
    assert RankBasedMetricResults.key_from_string(metric.key) == RankBasedMetricKey(
        side=SIDE_BOTH, rank_type=RANK_REALISTIC, metric=metric.key
    )


def test_sample_negatives():
    """Test for sample_negatives."""
    dataset = Nations()
    num_negatives = 2
    assert dataset.validation is not None
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
    for i, negatives in zip((0, 2), (head_negatives, tail_negatives), strict=False):
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


def test_sample_negatives_seed():
    """Test that sample_negatives is reproducible for a fixed seed and varies across seeds."""
    dataset = Nations()
    assert dataset.validation is not None
    kwargs = {
        "evaluation_triples": dataset.validation.mapped_triples,
        "additional_filter_triples": dataset.training.mapped_triples,
        "num_entities": dataset.num_entities,
        "num_samples": 5,
    }
    first = sample_negatives(**kwargs, random_state=42)
    second = sample_negatives(**kwargs, random_state=42)
    other = sample_negatives(**kwargs, random_state=43)
    for side in (LABEL_HEAD, LABEL_TAIL):
        assert first[side].dtype == torch.long
        assert first[side].shape == (dataset.validation.num_triples, 5)
        assert torch.equal(first[side], second[side])
    assert any(not torch.equal(first[side], other[side]) for side in (LABEL_HEAD, LABEL_TAIL))


def test_sample_negatives_no_candidates():
    """Test that a key without any candidate negative raises a ValueError."""
    # all three entities are true tails for (h=0, r=0), so there is no tail candidate
    triples = torch.as_tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    with pytest.raises(ValueError, match="no candidate"):
        sample_negatives(evaluation_triples=triples, num_entities=3, num_samples=2, random_state=0)


def test_sample_negatives_few_candidates():
    """Test that fewer candidates than samples falls back to duplicates drawn from the candidates."""
    # for tail prediction of (h=0, r=0), entity 2 is the only candidate
    triples = torch.as_tensor([[0, 0, 0], [0, 0, 1]])
    negatives = sample_negatives(evaluation_triples=triples, num_entities=3, num_samples=4, random_state=0)
    assert negatives[LABEL_TAIL].shape == (2, 4)
    assert (negatives[LABEL_TAIL] == 2).all()


class CandidateSetSizeTests(unittest.TestCase):
    """Tests for candidate set size calculation."""

    def setUp(self) -> None:
        """Prepare the test data."""
        self.dataset = Nations()

    def _test_get_candidate_set_size(
        self,
        mapped_triples: MappedTriples,
        restrict_entities_to: Collection[int] | None,
        restrict_relations_to: Collection[int] | None,
        additional_filter_triples: MappedTriples | list[MappedTriples] | None,
        num_entities: int | None,
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
        assert isinstance(df, pd.DataFrame)
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
            np.testing.assert_array_equal(df["index"], np.arange(mapped_triples.shape[0]))
            np.testing.assert_array_equal(df[list(COLUMN_LABELS)].values, mapped_triples.numpy())
        for candidate_column in (f"{LABEL_HEAD}_candidates", f"{LABEL_TAIL}_candidates"):
            assert (df[candidate_column] >= 1).all()
            assert (df[candidate_column] <= self.dataset.num_entities).all()

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
        assert self.dataset.validation is not None
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            None,
            None,
            self.dataset.validation.mapped_triples,
            self.dataset.num_entities,
        )

    def test_multi_filter(self):
        """Test with multiple additional filter triples."""
        assert self.dataset.validation is not None
        self._test_get_candidate_set_size(
            self.dataset.training.mapped_triples,
            None,
            None,
            (self.dataset.validation.mapped_triples, self.dataset.testing.mapped_triples),
            self.dataset.num_entities,
        )

    def test_all(self):
        """Test with filtering restriction and entity count inference."""
        assert self.dataset.validation is not None
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
            np.testing.assert_array_equal(df[column], df2[column])

    def test_duplicate_evaluation_triples(self):
        """Test that duplicate evaluation triples are not counted twice as positives."""
        # entities 0..3; (0, 0, 1) occurs twice
        mapped_triples = torch.as_tensor([[0, 0, 1], [0, 0, 1], [0, 0, 2]])
        df = get_candidate_set_size(mapped_triples=mapped_triples, num_entities=4)
        # tails for key (0, 0): {1, 2} -> 4 - 2 + 1 = 3 candidates (incl. the true one) for every row
        assert df["tail_candidates"].tolist() == [3, 3, 3]
        # heads for key (0, 1) / (0, 2): {0} -> 4 - 1 + 1 = 4 candidates
        assert df["head_candidates"].tolist() == [4, 4, 4]

    def test_entity_ids_with_gaps(self):
        """Test that the inferred number of entities is the maximum ID plus one."""
        # only entities 0, 2, 5 occur -> 3 unique, but 6 IDs
        mapped_triples = torch.as_tensor([[0, 0, 5], [2, 0, 5]])
        df = get_candidate_set_size(mapped_triples=mapped_triples)
        # tails for (0, 0): {5} -> 6 - 1 + 1; tails for (2, 0): {5} -> 6 - 1 + 1
        assert df["tail_candidates"].tolist() == [6, 6]
        # heads for (0, 5): {0, 2} -> 6 - 2 + 1
        assert df["head_candidates"].tolist() == [5, 5]

    def test_naive_count(self):
        """Compare against a naive count: all non-positive corruptions plus the true entity."""
        mapped_triples = self.dataset.training.mapped_triples[:30]
        num_entities = self.dataset.num_entities
        known = {tuple(t) for t in mapped_triples.tolist()}
        df = get_candidate_set_size(mapped_triples=mapped_triples, num_entities=num_entities)
        for row in df.itertuples():
            h, r, t = row.head, row.relation, row.tail
            expected_head = 1 + sum((e, r, t) not in known for e in range(num_entities) if e != h)
            expected_tail = 1 + sum((h, r, e) not in known for e in range(num_entities) if e != t)
            assert row.head_candidates == expected_head
            assert row.tail_candidates == expected_tail

    def test_empty_entity_restriction(self):
        """Test that an empty entity restriction is not treated as no restriction."""
        mapped_triples = torch.as_tensor([[0, 0, 1], [1, 0, 2]])
        df = get_candidate_set_size(mapped_triples=mapped_triples, restrict_entities_to=set())
        assert len(df) == 0


class ExpectedMetricsTests(unittest.TestCase):
    """Tests for expected metrics."""

    def _iter_num_candidates(self) -> Iterable[tuple[np.ndarray, int]]:
        """Generate number of ranking candidate arrays of different shapes."""
        generator: np.random.Generator = np.random.default_rng(seed=42)
        # test different shapes
        for shape, total in (
            ((), 20),
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
                assert ehk == pytest.approx(1.0)

    def test_expected_hits_at_k_manual(self):
        """Test expected Hits@k, where some candidate set sizes are smaller than k, but not all."""
        metric = HitsAtK(k=10)
        assert metric.expected_value(num_candidates=np.asarray([5, 20])) == pytest.approx((1 + 0.5) / 2)


def test_prepare_filter_triples():
    """Tests for prepare_filter_triples."""
    dataset = Nations()
    assert dataset.validation is not None
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


def test_restrict_entities_to_collection_types():
    """Test that `restrict_entities_to` can be given as set, list, or tensor, with identical results."""
    dataset = Nations()
    model = FixedModel(triples_factory=dataset.training)
    entities = [0, 2, 3, 5, 7, 9]
    results = []
    for restrict in (set(entities), list(entities), torch.as_tensor(entities)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            result = RankBasedEvaluator(filtered=True).evaluate(
                model=model,
                mapped_triples=dataset.testing.mapped_triples,
                additional_filter_triples=[dataset.training.mapped_triples],
                restrict_entities_to=restrict,
                pre_filtered_triples=False,
                batch_size=8,
                slice_size=None,
                use_tqdm=False,
            )
        results.append(result.to_dict())
    assert results[0] == results[1] == results[2]


def test_optimal_batch_size_group_id():
    """Test the stability of the batch size cache key."""
    evaluator = RankBasedEvaluator()
    model = FixedModel(triples_factory=Nations().training)

    def make_kwargs(**overrides):
        kwargs: dict[str, Any] = {
            "evaluator": evaluator,
            "mapped_triples": torch.zeros(10, 3, dtype=torch.long),
            "batch_size": 4,
            "slice_size": 5,
            "progress_bar": None,
            "model": model,
            "all_pos_triples": torch.zeros(20, 3, dtype=torch.long),
        }
        kwargs.update(overrides)
        return kwargs

    key = _optimal_batch_size_group_id(make_kwargs())
    # freshly created, equal-shaped tensors share the key; batch/slice size are ignored
    assert key == _optimal_batch_size_group_id(make_kwargs(all_pos_triples=torch.ones(20, 3, dtype=torch.long)))
    assert key == _optimal_batch_size_group_id(make_kwargs(batch_size=1, slice_size=2))
    # different shapes, number of evaluation triples, or evaluator instances do not
    assert key != _optimal_batch_size_group_id(make_kwargs(all_pos_triples=torch.zeros(21, 3, dtype=torch.long)))
    assert key != _optimal_batch_size_group_id(make_kwargs(mapped_triples=torch.zeros(11, 3, dtype=torch.long)))
    assert key != _optimal_batch_size_group_id(make_kwargs(evaluator=RankBasedEvaluator()))


class RankBasedMetricResultTests(cases.MetricResultTestCase):
    """Tests for rank-based metric results."""

    cls = RankBasedMetricResults
    num_entities: int = 7
    num_triples: int = 13

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["data"] = RankBasedMetricResults.create_random().data
        return kwargs

    def _verify_flat_dict(self, flat_dict: Mapping[str, Any]):
        for metric_cls in rank_based_metric_resolver:
            metric = metric_cls()
            metric_name = metric.key
            assert any(metric_name in key for key in flat_dict), metric_name

    def test_monotonicity_in_rank_type(self):
        """Test monotonicity for different rank-types."""
        hits_prefixes = [
            "hits_at_",
            "z_hits_at_",
            "adjusted_hits_at_",
        ]
        self.instance: RankBasedMetricResults
        targets = {key.side for key in self.instance.data}
        for metric_name in {key.metric for key in self.instance.data}:
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
                values = np.asarray(
                    [
                        self.instance.get_metric(
                            name=RankBasedMetricKey(side=target, rank_type=rank_type, metric=metric_name)
                        )
                        for rank_type in (RANK_PESSIMISTIC, RANK_REALISTIC, RANK_OPTIMISTIC)
                    ]
                )
                for i, j in pairwise(exp_sort_indices):
                    assert values[i] <= values[j], metric_name

    def test_to_df(self):
        """Test to_df."""
        df = self.instance.to_df()
        assert isinstance(df, pd.DataFrame)


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
    skip_cls: ClassVar[Collection[type]] = {DummyMetricResults}


class EvaluatorMetaTestCase(unittest_templates.MetaTestCase):
    """Test for tests for evaluators."""

    base_cls = Evaluator
    base_test = cases.EvaluatorTestCase
    skip_cls: ClassVar[Collection[type]] = {
        mocks.MockEvaluator,
        DummyEvaluator,
    }
