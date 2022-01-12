# -*- coding: utf-8 -*-

"""Inductive evaluator."""

import gc
import logging
import timeit
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from textwrap import dedent
from typing import Any, Collection, Iterable, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.autonotebook import tqdm

from .evaluator import (
    Evaluator,
    MetricResults,
    create_dense_positive_mask_,
    create_sparse_positive_filter_,
    filter_scores_,
)
from .rank_based_evaluator import RankBasedEvaluator
from ..models import Model
from ..triples.triples_factory import TriplesFactory, restrict_triples
from ..triples.utils import get_entities, get_relations
from ..typing import MappedTriples, Mode
from ..utils import (
    format_relative_comparison,
    is_cuda_oom_error,
    is_cudnn_error,
    is_nonzero_larger_than_maxint_error,
    normalize_string,
    split_list_in_batches_iter,
)

__all__ = [
    "InductiveEvaluator",
]

logger = logging.getLogger(__name__)


@contextmanager
def optional_context_manager(condition, context_manager):
    if condition:
        with context_manager:
            yield context_manager
    else:
        yield


class InductiveEvaluator(RankBasedEvaluator):
    """
    Inductive version of the evaluator. Main differences:
    - Takes the triple factory argument - on which the evaluation will be executed
    - Takes the mode argument which will be sent to the scoring function
    """

    def __init__(self, eval_factory: TriplesFactory, mode: Mode, **kwargs):
        super().__init__(**kwargs)

        self.mode = mode
        self.eval_factory = eval_factory

    def evaluate(
        self,
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_tqdm: bool = True,
        tqdm_kwargs: Optional[Mapping[str, str]] = None,
        restrict_entities_to: Optional[torch.LongTensor] = None,
        do_time_consuming_checks: bool = True,
        additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    ) -> MetricResults:
        """Run :func:`pykeen.evaluation.evaluate` with this evaluator."""
        if batch_size is None and self.automatic_memory_optimization:
            # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
            if model.device.type == "cpu":
                logger.info(
                    "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                    "Therefore, the batch_size will be set to the default value.",
                )
            else:
                batch_size, slice_size = self.batch_and_slice(
                    model=model,
                    mapped_triples=mapped_triples,
                    additional_filter_triples=additional_filter_triples,
                    batch_size=batch_size,
                    device=device,
                    use_tqdm=False,
                    restrict_entities_to=restrict_entities_to,
                    do_time_consuming_checks=do_time_consuming_checks,
                )
                # The batch_size and slice_size should be accessible to outside objects for re-use, e.g. early stoppers.
                self.batch_size = batch_size
                self.slice_size = slice_size

                # Clear the ranks from the current evaluator
                self.finalize()

        rv = evaluate(
            model=model,
            additional_filter_triples=additional_filter_triples,
            mapped_triples=mapped_triples,
            evaluators=self,
            batch_size=batch_size,
            slice_size=slice_size,
            device=device,
            squeeze=True,
            use_tqdm=use_tqdm,
            tqdm_kwargs=tqdm_kwargs,
            restrict_entities_to=restrict_entities_to,
            do_time_consuming_checks=do_time_consuming_checks,
            mode=self.mode,  # NEW training mode
        )
        # Since squeeze is true, we can expect that evaluate returns a MetricResult, but we need to tell MyPy that
        return cast(MetricResults, rv)


def evaluate(
    model: Model,
    mapped_triples: MappedTriples,
    evaluators: Union[Evaluator, Collection[Evaluator]],
    mode: Mode,  # new
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    slice_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    squeeze: bool = True,
    use_tqdm: bool = True,
    tqdm_kwargs: Optional[Mapping[str, str]] = None,
    restrict_entities_to: Optional[Collection[int]] = None,
    restrict_relations_to: Optional[Collection[int]] = None,
    do_time_consuming_checks: bool = True,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    pre_filtered_triples: bool = True,
) -> Union[MetricResults, List[MetricResults]]:
    """Evaluate metrics for model on mapped triples.

    The model is used to predict scores for all tails and all heads for each triple. Subsequently, each abstract
    evaluator is applied to the scores, also receiving the batch itself (e.g. to compute entity-specific metrics).
    Thereby, the (potentially) expensive score computation against all entities is done only once. The metric evaluators
    are expected to maintain their own internal buffers. They are returned after running the evaluation, and should
    offer a possibility to extract some final metrics.

    :param model:
        The model to evaluate.
    :param mapped_triples:
        The triples on which to evaluate. The mapped triples should never contain inverse triples - these are created by
        the model class on the fly.
    :param evaluators:
        An evaluator or a list of evaluators working on batches of triples and corresponding scores.
    :param mode:
        Evaluation mode: "valid" or "test"
    :param only_size_probing:
        The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
    :param batch_size: >0
        A positive integer used as batch size. Generally chosen as large as possible. Defaults to 1 if None.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param device:
        The device on which the evaluation shall be run. If None is given, use the model's device.
    :param squeeze:
        Return a single instance of :class:`MetricResults` if only one evaluator was given.
    :param use_tqdm:
        Should a progress bar be displayed?
    :param tqdm_kwargs:
        Additional keyword based arguments passed to the progress bar.
    :param restrict_entities_to:
        Optionally restrict the evaluation to the given entity IDs. This may be useful if one is only interested in a
        part of the entities, e.g. due to type constraints, but wants to train on all available data. For ranking the
        entities, we still compute all scores for all possible replacement entities to avoid irregular access patterns
        which might decrease performance, but the scores will afterwards be filtered to only keep those of interest.
        If provided, we assume by default that the triples are already filtered, such that it only contains the
        entities of interest. To explicitly filter within this method, pass `pre_filtered_triples=False`.
    :param restrict_relations_to:
        Optionally restrict the evaluation to the given relation IDs. This may be useful if one is only interested in a
        part of the relations, e.g. due to relation types, but wants to train on all available data. If provided, we
        assume by default that the triples are already filtered, such that it only contains the relations of interest.
        To explicitly filter within this method, pass `pre_filtered_triples=False`.
    :param do_time_consuming_checks:
        Whether to perform some time consuming checks on the provided arguments. Currently, this encompasses:
        - If restrict_entities_to or restrict_relations_to is not None, check whether the triples have been filtered.
        Disabling this option can accelerate the method. Only effective if pre_filtered_triples is set to True.
    :param pre_filtered_triples:
        Whether the triples have been pre-filtered to adhere to restrict_entities_to / restrict_relations_to. When set
        to True, and the triples have *not* been filtered, the results may be invalid. Pre-filtering the triples
        accelerates this method, and is recommended when evaluating multiple times on the same set of triples.
    :param additional_filtered_triples:
        Additional true triples to filter out during filtered evaluation.
    """
    if isinstance(evaluators, Evaluator):  # upgrade a single evaluator to a list
        evaluators = [evaluators]

    start = timeit.default_timer()

    # verify that the triples have been filtered
    if pre_filtered_triples and do_time_consuming_checks:
        if restrict_entities_to is not None:
            present_entity_ids = get_entities(triples=mapped_triples)
            unwanted = present_entity_ids.difference(restrict_entities_to)
            if len(unwanted) > 0:
                raise ValueError(
                    f"mapped_triples contains IDs of entities which are not contained in restrict_entities_to:"
                    f"{unwanted}. This will invalidate the evaluation results.",
                )
        if restrict_relations_to is not None:
            present_relation_ids = get_relations(triples=mapped_triples)
            unwanted = present_relation_ids.difference(restrict_relations_to)
            if len(unwanted):
                raise ValueError(
                    f"mapped_triples contains IDs of relations which are not contained in restrict_relations_to:"
                    f"{unwanted}. This will invalidate the evaluation results.",
                )

    # Filter triples if necessary
    if not pre_filtered_triples and (restrict_entities_to is not None or restrict_relations_to is not None):
        old_num_triples = mapped_triples.shape[0]
        mapped_triples = restrict_triples(
            mapped_triples=mapped_triples,
            entities=restrict_entities_to,
            relations=restrict_relations_to,
        )
        logger.info(f"keeping {format_relative_comparison(mapped_triples.shape[0], old_num_triples)} triples.")

    # Send to device
    if device is not None:
        model = model.to(device)
    device = model.device

    # Ensure evaluation mode
    model.eval()

    # Split evaluators into those which need unfiltered results, and those which require filtered ones
    filtered_evaluators = list(filter(lambda e: e.filtered, evaluators))
    unfiltered_evaluators = list(filter(lambda e: not e.filtered, evaluators))

    # Check whether we need to be prepared for filtering
    filtering_necessary = len(filtered_evaluators) > 0

    # Check whether an evaluator needs access to the masks
    # This can only be an unfiltered evaluator.
    positive_masks_required = any(e.requires_positive_mask for e in unfiltered_evaluators)

    # Prepare for result filtering
    if filtering_necessary or positive_masks_required:
        if additional_filter_triples is None:
            logger.warning(
                dedent(
                    """\
                The filtered setting was enabled, but there were no `additional_filter_triples`
                given. This means you probably forgot to pass (at least) the training triples. Try:

                    additional_filter_triples=[dataset.training.mapped_triples]

                Or if you want to use the Bordes et al. (2013) approach to filtering, do:

                    additional_filter_triples=[
                        dataset.training.mapped_triples,
                        dataset.validation.mapped_triples,
                    ]
            """
                )
            )
            all_pos_triples = mapped_triples
        elif isinstance(additional_filter_triples, (list, tuple)):
            all_pos_triples = torch.cat([*additional_filter_triples, mapped_triples], dim=0)
        else:
            all_pos_triples = torch.cat([additional_filter_triples, mapped_triples], dim=0)
        all_pos_triples = all_pos_triples.to(device=device)
    else:
        all_pos_triples = None

    # Send tensors to device
    mapped_triples = mapped_triples.to(device=device)

    # Prepare batches
    if batch_size is None:
        # This should be a reasonable default size that works on most setups while being faster than batch_size=1
        batch_size = 32
        logger.info(f"No evaluation batch_size provided. Setting batch_size to '{batch_size}'.")
    batches = cast(Iterable[np.ndarray], split_list_in_batches_iter(input_list=mapped_triples, batch_size=batch_size))

    # Show progressbar
    num_triples = mapped_triples.shape[0]

    # Flag to check when to quit the size probing
    evaluated_once = False

    # Disable gradient tracking
    _tqdm_kwargs = dict(
        desc=f"Evaluating on {model.device}",
        total=num_triples,
        unit="triple",
        unit_scale=True,
        # Choosing no progress bar (use_tqdm=False) would still show the initial progress bar without disable=True
        disable=not use_tqdm,
    )
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)
    with optional_context_manager(use_tqdm, tqdm(**_tqdm_kwargs)) as progress_bar, torch.inference_mode():
        # batch-wise processing
        for batch in batches:
            batch_size = batch.shape[0]
            relation_filter = None
            for column in (0, 2):
                relation_filter = _evaluate_batch(
                    batch=batch,  # TODO fix typing
                    model=model,
                    column=column,
                    filtered_evaluators=filtered_evaluators,
                    unfiltered_evaluators=unfiltered_evaluators,
                    slice_size=slice_size,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                    restrict_entities_to=restrict_entities_to,
                    positive_masks_required=positive_masks_required,
                    filtering_necessary=filtering_necessary,
                    mode=mode,  # new
                )

            # If we only probe sizes we do not need more than one batch
            if only_size_probing and evaluated_once:
                break

            evaluated_once = True

            if use_tqdm:
                progress_bar.update(batch_size)

        # Finalize
        results = [evaluator.finalize() for evaluator in evaluators]

    stop = timeit.default_timer()
    if only_size_probing:
        logger.debug("Evaluation took %.2fs seconds", stop - start)
    else:
        logger.info("Evaluation took %.2fs seconds", stop - start)

    if squeeze and len(results) == 1:
        return results[0]

    return results


def _evaluate_batch(
    batch: MappedTriples,
    model: Model,
    column: int,
    mode: Mode,  # new
    filtered_evaluators: Collection[Evaluator],
    unfiltered_evaluators: Collection[Evaluator],
    slice_size: Optional[int],
    all_pos_triples: Optional[MappedTriples],
    relation_filter: Optional[torch.BoolTensor],
    restrict_entities_to: Optional[torch.LongTensor],
    positive_masks_required: bool,
    filtering_necessary: bool,
) -> torch.BoolTensor:
    """
    Evaluate batch for all head predictions(column=0), or all tail predictions (column=2).

    :param batch: shape: (batch_size, 3)
        The batch of currently evaluated triples.
    :param model:
        The model to evaluate.
    :param column:
        The column which to evaluate. Either 0 for head prediction, or 2 for tail prediction.
    :param mode:
        Evaluation mode: "valid" or "test"
    :param filtered_evaluators:
        The evaluators which work on filtered scores.
    :param unfiltered_evaluators:
        The evaluators which work on unfiltered scores.
    :param slice_size:
        An optional slice size for computing the scores.
    :param all_pos_triples:
        All positive triples (required if filtering is necessary).
    :param relation_filter:
        The relation filter. Can be re-used.
    :param restrict_entities_to:
        Restriction to evaluate only for these entities.
    :param positive_masks_required:
        Whether dense positive masks are required (by any unfiltered evaluator).
    :param filtering_necessary:
        Whether filtering is necessary.

    :return:
        The relation filter, which can be re-used for the same batch.
    """
    if column not in {0, 2}:
        raise ValueError(f"column must be either 0 or 2, but is column={column}")

    # Predict scores once
    if column == 2:  # tail scores
        batch_scores_of_corrupted = model.predict_t(batch[:, 0:2], slice_size=slice_size, mode=mode)
    else:
        batch_scores_of_corrupted = model.predict_h(batch[:, 1:3], slice_size=slice_size, mode=mode)

    # Select scores of true
    batch_scores_of_true = batch_scores_of_corrupted[
        torch.arange(0, batch.shape[0]),
        batch[:, column],
    ]

    # Create positive filter for all corrupted
    if filtering_necessary or positive_masks_required:
        # Needs all positive triples
        if all_pos_triples is None:
            raise ValueError(
                "If filtering_necessary of positive_masks_required is True, all_pos_triples has to be "
                "provided, but is None."
            )

        # Create filter
        positive_filter, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=column,
        )

    # Create a positive mask with the size of the scores from the positive filter
    if positive_masks_required:
        positive_mask = create_dense_positive_mask_(
            zero_tensor=torch.zeros_like(batch_scores_of_corrupted),
            filter_batch=positive_filter,
        )
    else:
        positive_mask = None

    # Restrict to entities of interest
    if restrict_entities_to is not None:
        batch_scores_of_corrupted_ = batch_scores_of_corrupted[:, restrict_entities_to]
        positive_mask = positive_mask[:, restrict_entities_to]
    else:
        batch_scores_of_corrupted_ = batch_scores_of_corrupted

    # Evaluate metrics on these *unfiltered* scores
    for unfiltered_evaluator in unfiltered_evaluators:
        if column == 2:  # tail scores
            process = unfiltered_evaluator.process_tail_scores_
        else:
            process = unfiltered_evaluator.process_head_scores_
        process(
            hrt_batch=batch,
            true_scores=batch_scores_of_true[:, None],
            scores=batch_scores_of_corrupted_,
            dense_positive_mask=positive_mask,
        )

    # Filter
    if filtering_necessary:
        batch_filtered_scores_of_corrupted = filter_scores_(
            scores=batch_scores_of_corrupted,
            filter_batch=positive_filter,
        )

        # The scores for the true triples have to be rewritten to the scores tensor
        batch_filtered_scores_of_corrupted[
            torch.arange(0, batch.shape[0]),
            batch[:, column],
        ] = batch_scores_of_true

        # Restrict to entities of interest
        if restrict_entities_to is not None:
            batch_filtered_scores_of_corrupted = batch_filtered_scores_of_corrupted[:, restrict_entities_to]

        # Evaluate metrics on these *filtered* scores
        for filtered_evaluator in filtered_evaluators:
            if column == 2:  # tail scores
                process = filtered_evaluator.process_tail_scores_
            else:
                process = filtered_evaluator.process_head_scores_
            process(
                hrt_batch=batch,
                true_scores=batch_scores_of_true[:, None],
                scores=batch_filtered_scores_of_corrupted,
            )

    return relation_filter
