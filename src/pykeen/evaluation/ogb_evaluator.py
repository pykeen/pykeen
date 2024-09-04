"""OGB tools."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import torch
from torch_max_mem import maximize_memory_utilization
from tqdm.auto import tqdm

from .evaluator import MetricResults
from .rank_based_evaluator import RankBasedMetricKey, RankBasedMetricResults, SampledRankBasedEvaluator
from ..metrics import RankBasedMetric
from ..metrics.ranking import HitsAtK, InverseHarmonicMeanRank
from ..models import Model
from ..typing import LABEL_HEAD, LABEL_TAIL, RANK_REALISTIC, SIDE_BOTH, ExtendedTarget, MappedTriples, Target

__all__ = [
    "OGBEvaluator",
    "evaluate_ogb",
]

logger = logging.getLogger(__name__)


class OGBEvaluator(SampledRankBasedEvaluator):
    """A sampled, rank-based evaluator that applies a custom OGB evaluation."""

    # docstr-coverage: inherited
    def __init__(self, filtered: bool = False, **kwargs):  # noqa:D107
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
        batch_size: int | None = None,
        slice_size: int | None = None,
        device: torch.device | None = None,
        use_tqdm: bool = True,
        tqdm_kwargs: Mapping[str, str] | None = None,
        restrict_entities_to: Collection[int] | None = None,
        restrict_relations_to: Collection[int] | None = None,
        do_time_consuming_checks: bool = True,
        additional_filter_triples: None | MappedTriples | list[MappedTriples] = None,
        pre_filtered_triples: bool = True,
        targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
    ) -> MetricResults:
        """Run :func:`evaluate_ogb` with this evaluator."""
        if (
            {restrict_relations_to, restrict_entities_to, additional_filter_triples} != {None}
            or do_time_consuming_checks is False
            or pre_filtered_triples is False
        ):
            raise ValueError(
                f"{self} does not support any of {{restrict_relations_to, restrict_entities_to, "
                f"additional_filter_triples, do_time_consuming_checks, pre_filtered_triples}}",
            )
        return evaluate_ogb(
            evaluator=self,
            model=model,
            mapped_triples=mapped_triples,
            batch_size=batch_size,
            slice_size=slice_size,
            use_tqdm=use_tqdm,
            tqdm_kwargs=tqdm_kwargs,
            targets=targets,
        )


def evaluate_ogb(
    evaluator: SampledRankBasedEvaluator,
    model: Model,
    mapped_triples: MappedTriples,
    batch_size: int | None = None,
    slice_size: int | None = None,
    device: torch.device | None = None,
    use_tqdm: bool = True,
    tqdm_kwargs: Mapping[str, Any] | None = None,
    targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
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
    :param device:
            The device on which the evaluation shall be run. If None is given, use the model's device.

    :param batch_size:
        the batch size
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param use_tqdm:
        Should a progress bar be displayed?
    :param tqdm_kwargs:
        Additional keyword based arguments passed to the progress bar.
    :param targets:
        the prediction targets

    :return:
        the evaluation results

    :raises ImportError:
        if ogb is not installed
    :raises ValueError:
        if illegal ``additional_filter_triples`` argument is given in the kwargs
    """
    try:
        import ogb.linkproppred
    except ImportError as error:
        raise ImportError("OGB evaluation requires `ogb` to be installed.") from error

    # delay declaration
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

    # check targets
    if not set(targets).issubset(evaluator.negative_samples.keys()):
        raise ValueError(
            f"{targets=} are not supported by {evaluator=}, which only provides negative samples for "
            f"{sorted(evaluator.negative_samples.keys())}",
        )

    # filter supported metrics
    metrics: list[RankBasedMetric] = []
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
    y_pred_pos: dict[Target, torch.Tensor] = {}
    y_pred_neg: dict[Target, torch.Tensor] = {}

    # move tensor to device
    device = device or model.device
    model = model.to(device)
    mapped_triples = mapped_triples.to(device)

    # iterate over prediction targets
    tqdm_kwargs = dict(tqdm_kwargs or {})
    tqdm_kwargs["disable"] = not use_tqdm
    for target, negatives in evaluator.negative_samples.items():
        negatives = negatives.to(device)
        with tqdm(**tqdm_kwargs) as progress_bar:
            y_pred_pos[target], y_pred_neg[target] = _evaluate_ogb(
                evaluator=evaluator,
                batch_size=batch_size,
                slice_size=slice_size or model.num_entities,  # OGB evaluator supports head/tail only
                mapped_triples=mapped_triples,
                model=model,
                negatives=negatives,
                target=target,
                progress_bar=progress_bar,
            )

    def iter_preds() -> Iterable[tuple[ExtendedTarget, torch.Tensor, torch.Tensor]]:
        """Iterate over predicted scores for extended prediction targets."""
        targets = sorted(y_pred_pos.keys())
        for _target in targets:
            yield _target, y_pred_pos[_target], y_pred_neg[_target]
        yield (
            SIDE_BOTH,
            torch.cat([y_pred_pos[t] for t in targets], dim=0),
            torch.cat([y_pred_neg[t] for t in targets], dim=0),
        )

    result: dict[RankBasedMetricKey | str, float] = {}
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
            key = RankBasedMetricResults.key_from_string(key.replace("_list", "")).metric
            # OGB does not aggregate values across triples
            value = value.mean().item()
            result[RankBasedMetricKey(side=ext_target, rank_type=rank_type, metric=key)] = value
    return RankBasedMetricResults(data=result)


def _hasher(kwargs: Mapping[str, Any]) -> int:
    return hash((id(kwargs["model"]), kwargs["mapped_triples"].shape[0], kwargs["negatives"].shape, kwargs["target"]))


@maximize_memory_utilization(parameter_name=("batch_size", "slice_size"), hasher=_hasher)
def _evaluate_ogb(
    *,
    evaluator: OGBEvaluator,
    batch_size: int,
    slice_size: int,
    mapped_triples: MappedTriples,
    model: Model,
    negatives: torch.LongTensor,
    target: Target,
    progress_bar: tqdm,
) -> tuple[torch.Tensor, torch.Tensor]:
    # todo: maybe we can merge this code with the AMO code of the base evaluator?
    num_triples = mapped_triples.shape[0]
    progress_bar.reset(total=num_triples)
    # pre-allocate
    # TODO: maybe we want to collect scores on CPU / add an option?
    device = model.device
    y_pred_pos_side = torch.empty(size=(num_triples,), device=device)
    num_negatives = negatives.shape[1]
    y_pred_neg_side = torch.empty(size=(num_triples, num_negatives), device=device)
    # iterate over batches
    offset = 0
    for hrt_batch, negatives_batch in zip(
        mapped_triples.split(split_size=batch_size), negatives.split(split_size=batch_size)
    ):
        # combine ids, shape: (batch_size, num_negatives + 1)
        ids = torch.cat([hrt_batch[:, 2, None], negatives_batch], dim=1)
        # get scores, shape: (batch_size, num_negatives + 1)
        scores = model.predict(hrt_batch=hrt_batch, target=target, ids=ids, mode=evaluator.mode, slice_size=slice_size)
        # store positive and negative scores
        this_batch_size = scores.shape[0]
        stop = offset + this_batch_size
        y_pred_pos_side[offset:stop] = scores[:, 0]
        y_pred_neg_side[offset:stop] = scores[:, 1:]
        offset = stop
        progress_bar.update(hrt_batch.shape[0])
    return y_pred_pos_side, y_pred_neg_side
