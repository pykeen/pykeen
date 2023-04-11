"""OGB tools."""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .evaluator import MetricResults
from .rank_based_evaluator import RankBasedMetricResults, SampledRankBasedEvaluator
from .ranking_metric_lookup import MetricKey
from ..metrics import RankBasedMetric
from ..metrics.ranking import HitsAtK, InverseHarmonicMeanRank
from ..models import Model
from ..typing import RANK_REALISTIC, SIDE_BOTH, ExtendedTarget, MappedTriples, RankType, Target

__all__ = [
    "OGBEvaluator",
    "evaluate_ogb",
]

logger = logging.getLogger(__name__)


class OGBEvaluator(SampledRankBasedEvaluator):
    """A sampled, rank-based evaluator that applies a custom OGB evaluation."""

    # docstr-coverage: inherited
    def __init__(self, filtered: bool = False, **kwargs):
        if filtered:
            raise ValueError(
                "OGB evaluator is already filtered, but not dynamically like other evaluators because "
                "it requires pre-calculated filtered negative triples. Therefore, it is not allowed to "
                "accept filtered=True"
            )
        super().__init__(**kwargs, filtered=filtered)

    def evaluate(
        self,
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        **kwargs,
    ) -> MetricResults:
        """Run :func:`evaluate_ogb` with this evaluator."""
        return evaluate_ogb(
            evaluator=self,
            model=model,
            mapped_triples=mapped_triples,
            batch_size=batch_size,
            **kwargs,
        )


def evaluate_ogb(
    evaluator: SampledRankBasedEvaluator,
    model: Model,
    mapped_triples: MappedTriples,
    batch_size: Optional[int] = None,
    **kwargs,
) -> MetricResults:
    """
    Evaluate a model using OGB's evaluator.

    :param evaluator:
        An evaluator
    :param model:
        the model; will be set to evaluation mode.
    :param mapped_triples:
        the evaluation triples

        .. note ::
            the evaluation triples have to match with the stored explicit negatives

    :param batch_size:
        the batch size
    :param kwargs:
        additional keyword-based parameters passed to :meth:`pykeen.nn.Model.predict`

    :return:
        the evaluation results

    :raises ImportError:
        if ogb is not installed
    :raises NotImplementedError:
        if `batch_size` is None, i.e., automatic batch size selection is selected
    :raises ValueError:
        if illegal ``additional_filter_triples`` argument is given in the kwargs
    """
    try:
        import ogb.linkproppred
    except ImportError as error:
        raise ImportError("OGB evaluation requires `ogb` to be installed.") from error

    if batch_size is None:
        raise NotImplementedError("Automatic batch size selection not available for OGB evaluation.")

    additional_filter_triples = kwargs.pop("additional_filter_triples", None)
    if additional_filter_triples is not None:
        raise ValueError(
            f"evaluate_ogb received additional_filter_triples={additional_filter_triples}. However, it uses "
            f"explicitly given filtered negative triples, and therefore shouldn't be passed any additional ones"
        )

    class _OGBEvaluatorBridge(ogb.linkproppred.Evaluator):
        """A wrapper around OGB's evaluator to support evaluation on non-OGB datasets."""

        def __init__(self):
            """Initialize the evaluator."""
            # note: OGB's evaluator needs a dataset name as input, and uses it to lookup the standard evaluation
            # metric. we do want to support user-selected metrics on arbitrary datasets instead

    ogb_evaluator = _OGBEvaluatorBridge()
    # this setting is equivalent to the WikiKG2 setting, and will calculate MRR *and* H@k for k in {1, 3, 10}
    ogb_evaluator.eval_metric = "mrr"
    ogb_evaluator.K = None

    # filter supported metrics
    metrics: List[RankBasedMetric] = []
    for metric in evaluator.metrics:
        if not isinstance(metric, (HitsAtK, InverseHarmonicMeanRank)) or (
            isinstance(metric, HitsAtK) and metric.k not in {1, 3, 10}
        ):
            logger.warning(f"{metric} is not supported by OGB evaluator")
            continue
        metrics.append(metric)

    # prepare input format, cf. `evaluator.expected_input``
    # y_pred_pos: shape: (num_edge,)
    # y_pred_neg: shape: (num_edge, num_nodes_neg)
    y_pred_pos: Dict[Target, torch.Tensor] = {}
    y_pred_neg: Dict[Target, torch.Tensor] = {}

    num_triples = mapped_triples.shape[0]
    device = mapped_triples.device
    # iterate over prediction targets
    for target, negatives in evaluator.negative_samples.items():
        # pre-allocate
        # TODO: maybe we want to collect scores on CPU / add an option?
        y_pred_pos[target] = y_pred_pos_side = torch.empty(size=(num_triples,), device=device)
        num_negatives = negatives.shape[1]
        y_pred_neg[target] = y_pred_neg_side = torch.empty(size=(num_triples, num_negatives), device=device)
        # iterate over batches
        offset = 0
        for hrt_batch, negatives_batch in zip(
            mapped_triples.split(split_size=batch_size), negatives.split(split_size=batch_size)
        ):
            # combine ids, shape: (batch_size, num_negatives + 1)
            ids = torch.cat([hrt_batch[:, 2, None], negatives_batch], dim=1)
            # get scores, shape: (batch_size, num_negatives + 1)
            scores = model.predict(hrt_batch=hrt_batch, target=target, ids=ids, mode=evaluator.mode, **kwargs)
            # store positive and negative scores
            this_batch_size = scores.shape[0]
            stop = offset + this_batch_size
            y_pred_pos_side[offset:stop] = scores[:, 0]
            y_pred_neg_side[offset:stop] = scores[:, 1:]
            offset = stop

    def iter_preds() -> Iterable[Tuple[ExtendedTarget, torch.Tensor, torch.Tensor]]:
        """Iterate over predicted scores for extended prediction targets."""
        targets = sorted(y_pred_pos.keys())
        for _target in targets:
            yield _target, y_pred_pos[_target], y_pred_neg[_target]
        yield (
            SIDE_BOTH,
            torch.cat([y_pred_pos[t] for t in targets], dim=0),
            torch.cat([y_pred_neg[t] for t in targets], dim=0),
        )

    result: Dict[Tuple[str, ExtendedTarget, RankType], float] = {}
    # cf. https://github.com/snap-stanford/ogb/pull/357
    rank_type = RANK_REALISTIC
    for ext_target, y_pred_pos_side, y_pred_neg_side in iter_preds():
        # combine to input dictionary
        input_dict = dict(y_pred_pos=y_pred_pos_side, y_pred_neg=y_pred_neg_side)
        # delegate to OGB evaluator
        ogb_result = ogb_evaluator.eval(input_dict=input_dict)
        # post-processing
        for key, value in ogb_result.items():
            # normalize name
            key = MetricKey.lookup(key.replace("_list", "")).metric
            # OGB does not aggregate values across triples
            value = value.mean().item()
            result[key, ext_target, rank_type] = value
    return RankBasedMetricResults(data=result)
