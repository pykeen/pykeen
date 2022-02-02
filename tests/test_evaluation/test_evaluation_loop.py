from typing import Any, MutableMapping
import pykeen.evaluation.evaluation_loop
import pykeen.evaluation.rank_based_evaluator
from tests import cases


class LinkPredictionEvaluationLoopTestCase(cases.EvaluationLoopTestCase):
    """Test the link prediction evaluation loop."""

    cls = pykeen.evaluation.evaluation_loop.LinkPredictionEvaluationLoop

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["triples_factory"] = self.factory
        kwargs["evaluator"] = pykeen.evaluation.rank_based_evaluator.RankBasedEvaluator(filtered=False)
        return kwargs
