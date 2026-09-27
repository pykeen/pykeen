"""Tests for evaluation loops."""

from collections.abc import MutableMapping
from typing import Any

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
    kwargs = {"targets": (LABEL_RELATION,)}


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
