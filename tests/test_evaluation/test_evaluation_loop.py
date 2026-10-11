"""Tests for evaluation loops."""

from collections.abc import Mapping, MutableMapping
from typing import Any, ClassVar

import pytest
import torch

import pykeen.evaluation.evaluation_loop
import pykeen.evaluation.rank_based_evaluator
from pykeen.datasets import Nations
from pykeen.evaluation import Evaluator, RankBasedEvaluator
from pykeen.evaluation.classification_evaluator import ClassificationEvaluator
from pykeen.evaluation.evaluation_loop import LCWAEvaluationDataset, LCWAEvaluationLoop
from pykeen.models import FixedModel
from pykeen.typing import LABEL_RELATION
from tests import cases


class LinkPredictionEvaluationLoopTestCase(cases.EvaluationLoopTestCase):
    """Test the link prediction evaluation loop."""

    cls = pykeen.evaluation.evaluation_loop.LCWAEvaluationLoop

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["triples_factory"] = self.factory
        return kwargs


class RelationPredictionLinkPredictionEvaluationLoopTestCase(LinkPredictionEvaluationLoopTestCase):
    """Test the link prediction evaluation loop for relation prediction."""

    cls = pykeen.evaluation.evaluation_loop.LCWAEvaluationLoop
    kwargs: ClassVar[Mapping[str, Any]] = {"targets": (LABEL_RELATION,)}


@pytest.mark.parametrize(
    "evaluator",
    [
        RankBasedEvaluator(filtered=True),
        # the raw / unfiltered ranking protocol has to work, too
        RankBasedEvaluator(filtered=False),
        # exercises the dense positive mask code path
        ClassificationEvaluator(),
    ],
)
def test_loop_matches_evaluate(evaluator: Evaluator):
    """Test that the LCWA evaluation loop agrees with :meth:`Evaluator.evaluate`."""
    dataset = Nations()
    model = FixedModel(triples_factory=dataset.training)
    # note: the filter triples are ignored by both paths if neither filtering nor masks are required
    additional_filter_triples = [dataset.training.mapped_triples]
    loop_result = LCWAEvaluationLoop(
        model=model,
        triples_factory=dataset.testing,
        evaluator=evaluator,
        additional_filter_triples=additional_filter_triples,
    ).evaluate(batch_size=8, use_tqdm=False)
    result = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=additional_filter_triples,
        batch_size=8,
        use_tqdm=False,
    )
    assert loop_result.to_flat_dict() == pytest.approx(result.to_flat_dict())


@torch.inference_mode()
def test_process_batch_filtered_and_masked():
    """Test that filtering and the dense positive masks are not mutually exclusive."""
    dataset = Nations()
    # note: no in-tree evaluator requires both, but the combination is supported by Evaluator.evaluate
    evaluator = ClassificationEvaluator()
    evaluator.filtered = True
    loop = LCWAEvaluationLoop(
        model=FixedModel(triples_factory=dataset.training),
        triples_factory=dataset.testing,
        evaluator=evaluator,
        additional_filter_triples=[dataset.training.mapped_triples],
    )
    # note: the classification evaluator raises if the dense positive mask is missing
    loop.process_batch(batch=next(iter(loop.get_loader(batch_size=2))))
    assert evaluator.all_positives


@pytest.mark.parametrize("as_sequence", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_dataset_additional_filter_triples(filtered: bool, as_sequence: bool):
    """Test that a single tensor is accepted as additional filter triples."""
    dataset = Nations()
    additional_filter_triples = dataset.training.mapped_triples
    LCWAEvaluationDataset(
        factory=dataset.testing,
        filtered=filtered,
        additional_filter_triples=[additional_filter_triples] if as_sequence else additional_filter_triples,
    )


class _OOMOnceModel(FixedModel):
    """A model which raises an out-of-memory error on one specific call of :meth:`predict`."""

    def __init__(self, fail_on_call: int, **kwargs):
        """Initialize the model.

        :param fail_on_call: the (1-based) index of the call to :meth:`predict` which fails, exactly once
        :param kwargs: additional keyword-based parameters passed to :meth:`FixedModel.__init__`
        """
        super().__init__(**kwargs)
        self.fail_on_call = fail_on_call
        self.num_calls = 0
        self.num_failures = 0

    def predict(self, *args, **kwargs):
        self.num_calls += 1
        if self.num_calls == self.fail_on_call:
            self.num_failures += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return super().predict(*args, **kwargs)


def _num_ranking_tasks(evaluator: RankBasedEvaluator) -> int:
    """Count the ranks collected by the evaluator, across all targets (using the realistic rank type)."""
    return sum(
        len(chunk) for (_, rank_type), chunks in evaluator.ranks.items() if rank_type == "realistic" for chunk in chunks
    )


@pytest.mark.parametrize("use_loop", [False, True])
def test_oom_retry_does_not_double_count(use_loop: bool):
    """Test that batches processed before an out-of-memory error are not counted again after the retry."""
    dataset = Nations()
    num_triples = dataset.testing.num_triples

    def _run(model: FixedModel) -> tuple[RankBasedEvaluator, dict[str, float]]:
        evaluator = RankBasedEvaluator(filtered=False, clear_on_finalize=False)
        if use_loop:
            result = LCWAEvaluationLoop(model=model, triples_factory=dataset.testing, evaluator=evaluator).evaluate(
                batch_size=64, use_tqdm=False
            )
        else:
            result = evaluator.evaluate(
                model=model, mapped_triples=dataset.testing.mapped_triples, batch_size=64, use_tqdm=False
            )
        return evaluator, result.to_flat_dict()

    reference_evaluator, reference = _run(FixedModel(triples_factory=dataset.training))
    model = _OOMOnceModel(fail_on_call=3, triples_factory=dataset.training)
    evaluator, result = _run(model)

    # sanity check: the fake error was raised, and evaluation continued afterwards
    assert model.num_failures == 1
    assert model.num_calls > 3
    # each test triple is ranked once for head and once for tail prediction
    assert _num_ranking_tasks(reference_evaluator) == 2 * num_triples
    assert _num_ranking_tasks(evaluator) == 2 * num_triples
    assert result == pytest.approx(reference)
