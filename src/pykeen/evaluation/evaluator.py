# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

import gc
import logging
import timeit
from abc import ABC, abstractmethod
from contextlib import contextmanager
from math import ceil
from typing import Any, ClassVar, Collection, List, Mapping, Optional, Tuple, Type, Union, cast

import pandas
import torch
from tqdm.autonotebook import tqdm

from ..constants import COLUMN_LABELS, TARGET_TO_INDEX, TARGET_TO_KEY_LABELS
from ..metrics.utils import Metric
from ..models import Model
from ..triples.triples_factory import restrict_triples
from ..triples.utils import get_entities, get_relations
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, InductiveMode, MappedTriples, Target
from ..utils import (
    format_relative_comparison,
    is_cuda_oom_error,
    is_cudnn_error,
    is_nonzero_larger_than_maxint_error,
    normalize_string,
    prepare_filter_triples,
)

__all__ = [
    "Evaluator",
    "MetricResults",
    "filter_scores_",
    "evaluate",
    "prepare_filter_triples",
]

logger = logging.getLogger(__name__)


@contextmanager
def optional_context_manager(condition, context_manager):
    """Return an optional context manager based on the given condition."""
    if condition:
        with context_manager:
            yield context_manager
    else:
        yield


class MetricResults:
    """Results from computing metrics."""

    metrics: ClassVar[Mapping[str, Type[Metric]]]

    def __init__(self, data: Mapping):
        """Initialize the result wrapper."""
        self.data = data

    def __getattr__(self, item):  # noqa:D105
        # TODO remove this, it makes code much harder to reason about
        if item not in self.data:
            raise AttributeError
        return self.data[item]

    @abstractmethod
    def get_metric(self, name: str) -> float:
        """Get the given metric from the results.

        :param name: The name of the metric
        :returns: The value for the metric
        """
        raise NotImplementedError

    def to_dict(self):
        """Get the results as a dictionary."""
        return self.data

    def to_flat_dict(self) -> Mapping[str, Any]:
        """Get the results as a flattened dictionary."""
        return self.to_dict()


class Evaluator(ABC):
    """An abstract evaluator for KGE models.

    The evaluator encapsulates the computation of evaluation metrics based on head and tail scores. To this end, it
    offers two methods to process a batch of triples together with the scores produced by some model. It maintains
    intermediate results in its state, and offers a method to obtain the final results once finished.
    """

    def __init__(
        self,
        filtered: bool = False,
        requires_positive_mask: bool = False,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        automatic_memory_optimization: bool = True,
        mode: Optional[InductiveMode] = None,
    ):
        """Initialize the evaluator.

        :param filtered: Should filtered evaluation be performed?
        :param requires_positive_mask: Does the evaluator need access to the masks?
        :param batch_size: >0. Evaluation batch size.
        :param slice_size: >0. The divisor for the scoring function when using slicing
        :param automatic_memory_optimization: Whether to automatically optimize the sub-batch size during
            evaluation with regards to the hardware at hand.
        :param mode:
            the inductive mode, or None for transductive evaluation
        """
        self.filtered = filtered
        self.requires_positive_mask = requires_positive_mask
        self.batch_size = batch_size
        self.slice_size = slice_size
        self.automatic_memory_optimization = automatic_memory_optimization
        self.mode = mode

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the evaluator."""
        return normalize_string(cls.__name__, suffix=Evaluator.__name__)

    @abstractmethod
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
        """Process a batch of triples with their computed scores for all entities.

        :param hrt_batch: shape: (batch_size, 3)
        :param target:
            the prediction target
        :param scores: shape: (batch_size, num_entities)
        :param true_scores: shape: (batch_size, 1)
        :param dense_positive_mask: shape: (batch_size, num_entities)
            An optional binary (0/1) tensor indicating other true entities.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear buffers and intermediate results."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> MetricResults:
        """Compute the final results, and clear buffers."""
        raise NotImplementedError

    def evaluate(
        self,
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        **kwargs,
    ) -> MetricResults:
        """
        Run :func:`pykeen.evaluation.evaluate` with this evaluator.

        This method will re-use the stored optimized batch and slice size, as well as the evaluator's inductive mode.

        :param model:
            the model to evaluate.
        :param mapped_triples: shape: (n, 3)
            the ID-based evaluation triples
        :param batch_size:
            the batch size to use, or `None` to trigger automatic memory optimization
        :param slice_size:
            the slice size to use
        :param kwargs:
            the keyword-based parameters passed to :func:`pykeen.evaluation.evaluate`

        :return:
            the evaluation results
        """
        # add mode parameter
        mode = kwargs.pop("mode", None)
        if mode is not None:
            logger.warning(f"Ignoring provided mode={mode}, and use the evaluator's mode={self.mode} instead")
        kwargs["mode"] = self.mode

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
                    batch_size=batch_size,
                    **kwargs,
                )
                # The batch_size and slice_size should be accessible to outside objects for re-use, e.g. early stoppers.
                self.batch_size = batch_size
                self.slice_size = slice_size

                # Clear the ranks from the current evaluator
                self.clear()

        rv = evaluate(
            model=model,
            mapped_triples=mapped_triples,
            evaluator=self,
            batch_size=batch_size,
            slice_size=slice_size,
            **kwargs,
        )
        # Since squeeze is true, we can expect that evaluate returns a MetricResult, but we need to tell MyPy that
        return cast(MetricResults, rv)

    def batch_and_slice(
        self,
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[int, Optional[int]]:
        """Find the maximum possible batch_size and slice_size for evaluation with the current setting.

        The speed of evaluation can be greatly increased when the batch_size is increased, therefore this function
        estimates the maximal possible batch_size for the evaluation by starting with the batch_size given as argument
        and increasing it until the hardware runs out-of-memory(OOM). In some cases, i.e. with very large models or very
        large datasets, even the batch_size 1 is too big for the hardware at hand. In these cases, this function will
        check if the model at hand allows slicing (this needs to be implemented for the affected scoring functions) and,
        if possible, will search the maximum possible slice_size that would still allow to calculate the model with the
        given parameters on the hardware at hand.

        :param model:
            The model to evaluate.
        :param mapped_triples:
            The triples on which to evaluate.
        :param batch_size:
            The initial batch size to start with. None defaults to number_of_triples.
        :param kwargs:
            additional keyword-based parameters passed to :func:`pykeen.evaluation.evaluate`

        :return:
            Maximum possible batch size and, if necessary, the slice_size, which defaults to None.

        :raises MemoryError:
            If it is not possible to evaluate the model on the hardware at hand with the given parameters.
        """
        # do not display progress bar while searching
        kwargs["use_tqdm"] = False
        # start by searching for batch_size
        batch_size, evaluated_once = self._param_size_search(
            key="batch_size",
            start_value=batch_size,
            model=model,
            mapped_triples=mapped_triples,
            **kwargs,
        )

        if evaluated_once:  # slice_size = None
            return batch_size, None

        # We need to try slicing, if the evaluation for the batch_size search never succeeded
        # we do not need to repeat time-consuming checks
        kwargs["do_time_consuming_checks"] = False
        slice_size, evaluated_once = self._param_size_search(
            key="slice_size",
            # infer start value
            start_value=None,
            model=model,
            mapped_triples=mapped_triples,
            **kwargs,
        )
        if not evaluated_once:
            raise MemoryError("The current model can't be trained on this hardware with these parameters.")

        return batch_size, slice_size

    def _param_size_search(
        self,
        key: str,
        start_value: Optional[int],
        model: Model,
        mapped_triples: MappedTriples,
        **kwargs,
    ) -> Tuple[int, bool]:
        values_dict = {}
        maximum_triples = mapped_triples.shape[0]
        if key == "batch_size":
            if start_value is None:
                start_value = 256
            if start_value > maximum_triples:
                start_value = maximum_triples
            values_dict[key] = start_value
            values_dict["slice_size"] = None
        elif key == "slice_size":
            if start_value is None:
                # Since the batch_size search with size 1, i.e. one tuple ((h, r), (h, t) or (r, t)) scored on all
                # entities/relations, must have failed to start slice_size search, we start with trying half the
                # entities/relations.
                targets: Collection[Target] = kwargs.get("targets", (LABEL_HEAD, LABEL_TAIL))
                predict_entities = bool({LABEL_HEAD, LABEL_TAIL}.intersection(targets))
                predict_relations = LABEL_RELATION in targets
                max_id = -1
                if predict_entities:
                    max_id = max(max_id, model.num_entities)
                if predict_relations:
                    max_id = max(max_id, model.num_relations)
                start_value = ceil(max_id / 2)
            self._check_slicing_availability(
                model, batch_size=1, entities=predict_entities, relations=predict_relations
            )
            values_dict[key] = start_value
            values_dict["batch_size"] = 1
        else:
            raise AttributeError(f"The parameter {key} is unknown.")

        reached_max = False
        evaluated_once = False
        logger.info(f"Starting {key} search for evaluation now...")
        while True:
            logger.debug(f"Trying {key}={values_dict[key]}")
            try:
                # The cache of the previous run has to be freed to allow accurate memory availability estimates
                gc.collect()
                torch.cuda.empty_cache()
                evaluate(
                    model=model,
                    mapped_triples=mapped_triples,
                    evaluator=self,
                    only_size_probing=True,
                    batch_size=values_dict.get("batch_size"),
                    slice_size=values_dict.get("slice_size"),
                    **kwargs,
                )
                evaluated_once = True
            except RuntimeError as runtime_error:
                # Due to the caused OOM Runtime Error, the failed model has to be cleared to avoid memory leakage
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                # The cache of the previous run has to be freed to allow accurate memory availability estimates
                gc.collect()
                torch.cuda.empty_cache()
                if (
                    not is_cudnn_error(runtime_error)
                    and not is_cuda_oom_error(runtime_error)
                    and not is_nonzero_larger_than_maxint_error(runtime_error)
                ):
                    raise runtime_error
                if values_dict[key] == 1:
                    logger.debug(
                        f"Even {key} {values_dict[key]} does not fit into your memory with these parameters.",
                    )
                    break

                #  values_dict[key] will always be an int at this point
                values_dict[key] //= 2  # type: ignore
                reached_max = True
                if evaluated_once:
                    logger.info(f"Concluded {key} search with batch_size={values_dict[key]}.")
                    break
                else:
                    logger.debug(f"The {key} {values_dict[key]} was too big, trying less now")
            else:
                # The cache of the previous run has to be freed to allow accurate memory availability estimates
                gc.collect()
                torch.cuda.empty_cache()
                if not reached_max and values_dict["batch_size"] < maximum_triples:
                    values_dict[key] *= 2  # type: ignore
                else:
                    logger.info(f"Concluded {key} search with batch_size={values_dict[key]}.")
                    break

        return cast(Tuple[int, bool], (values_dict[key], evaluated_once))

    @staticmethod
    def _check_slicing_availability(model: Model, batch_size: int, entities: bool, relations: bool) -> None:
        """
        Raise an error if the necessary slicing operations are not supported.

        :param model:
            the model
        :param batch_size:
            the batch-size; only used for creating the error message
        :param entities:
            whether entities need to be scored
        :param relations:
            whether relations need to be scored

        :raises MemoryError:
            if the necessary slicing operations are not supported by the model
        """
        reasons = []
        if entities:
            # if inverse triples are used, we only do score_t (TODO: by default; can this be changed?)
            if not model.can_slice_t:
                reasons.append("score_t")
            # otherwise, i.e., without inverse triples, we also need score_h
            if not model.use_inverse_triples and not model.can_slice_t:
                reasons.append("score_h")
        # if relations are to be predicted, we need to slice score_r
        if relations and not model.can_slice_r:
            reasons.append("score_r")
        # raise an error, if any of the required methods cannot slice
        if reasons:
            raise MemoryError(
                f"The current model can't be evaluated on this hardware with these parameters, as "
                f"evaluation batch_size={batch_size} is too big and slicing is not implemented for this "
                f"model yet (missing support for: {reasons})"
            )


def create_sparse_positive_filter_(
    hrt_batch: MappedTriples,
    all_pos_triples: torch.LongTensor,
    relation_filter: torch.BoolTensor = None,
    filter_col: int = 0,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """Compute indices of all positives.

    For simplicity, only the head-side is described, i.e. filter_col=0. The tail-side is processed alike.

    For each (h, r, t) triple in the batch, the entity identifiers are computed such that (h', r, t) exists in all
    positive triples.

    :param hrt_batch: shape: (batch_size, 3)
        A batch of triples.
    :param all_pos_triples: shape: (num_positive_triples, 3)
        All positive triples to base the filtering on.
    :param relation_filter: shape: (batch_size, num_positive_triples)
        A boolean mask R[i, j] which is True iff the j-th positive triple contains the same relation as the i-th triple
        in the batch.
    :param filter_col:
        The column along which to filter. Allowed are {0, 2}, where 0 corresponds to filtering head-based and 2
        corresponds to filtering tail-based.

    :return:
        - positives, shape: (2, m)
            The indices of positives in format [(batch_index, entity_id)].
        - the relation filter for re-usage.

    :raises NotImplementedError:
        if the `filter_col` is not in `{0, 2}`
    """
    if filter_col not in {0, 2}:
        raise NotImplementedError(
            "This code has only been written for updating head (filter_col=0) or "
            f"tail (filter_col=2) mask, but filter_col={filter_col} was given.",
        )

    if relation_filter is None:
        relations = hrt_batch[:, 1:2]
        relation_filter = (all_pos_triples[:, 1:2]).view(1, -1) == relations

    # Split batch
    other_col = 2 - filter_col
    entities = hrt_batch[:, other_col : other_col + 1]

    entity_filter_test = (all_pos_triples[:, other_col : other_col + 1]).view(1, -1) == entities
    filter_batch = (entity_filter_test & relation_filter).nonzero(as_tuple=False)
    filter_batch[:, 1] = all_pos_triples[:, filter_col : filter_col + 1].view(1, -1)[:, filter_batch[:, 1]]

    return filter_batch, relation_filter


def create_dense_positive_mask_(
    zero_tensor: torch.FloatTensor,
    filter_batch: torch.LongTensor,
) -> torch.FloatTensor:
    """Construct dense positive mask.

    :param zero_tensor: shape: (batch_size, num_entities)
        A tensor of zeros of suitable shape.
    :param filter_batch: shape: (m, 2)
        The indices of all positives in format (batch_index, entity_id)
    :return:
        The dense positive mask with x[b, i] = 1 iff (b, i) in filter_batch.
    """
    zero_tensor[filter_batch[:, 0], filter_batch[:, 1]] = 1

    return zero_tensor


def filter_scores_(
    scores: torch.FloatTensor,
    filter_batch: torch.LongTensor,
) -> torch.FloatTensor:
    """Filter scores by setting true scores to NaN.

    :param scores: shape: (batch_size, num_entities)
        The scores for all corrupted triples (including the currently considered true triple). Are modified *in-place*.
    :param filter_batch: (m, 2)
        The indices of all positives.

    :return:
        A reference to the scores, which have been updated in-place.
    """
    # Bind shape
    batch_size, num_entities = scores.shape

    # Set all filtered triples to NaN to ensure their exclusion in subsequent calculations
    scores[filter_batch[:, 0], filter_batch[:, 1]] = float("nan")

    # Warn if all entities will be filtered
    # (scores != scores) yields true for all NaN instances (IEEE 754), thus allowing to count the filtered triples.
    if ((scores != scores).sum(dim=1) == num_entities).any():
        logger.warning(
            "User selected filtered metric computation, but all corrupted triples exists also as positive " "triples",
        )

    return scores


# TODO: consider switching to torch.DataLoader where the preparation of masks/filter batches also takes place
def evaluate(
    model: Model,
    mapped_triples: MappedTriples,
    evaluator: Evaluator,
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    slice_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_tqdm: bool = True,
    tqdm_kwargs: Optional[Mapping[str, str]] = None,
    restrict_entities_to: Optional[Collection[int]] = None,
    restrict_relations_to: Optional[Collection[int]] = None,
    do_time_consuming_checks: bool = True,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    pre_filtered_triples: bool = True,
    targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
    *,
    mode: Optional[InductiveMode],
) -> MetricResults:
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
    :param evaluator:
        The evaluator.
    :param only_size_probing:
        The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
    :param batch_size: >0
        A positive integer used as batch size. Generally chosen as large as possible. Defaults to 1 if None.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param device:
        The device on which the evaluation shall be run. If None is given, use the model's device.
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
    :param additional_filter_triples:
        additional true triples to filter out during filtered evaluation.
    :param targets:
        the prediction targets
    :param mode:
        the inductive mode, or None for transductive evaluation

    :raises NotImplementedError:
        if relation prediction evaluation is requested
    :raises ValueError:
        if the pre_filtered_triples contain unwanted entities (can only be detected with the time-consuming checks).

    :return:
        the evaluation results
    """
    if LABEL_RELATION in targets:
        raise NotImplementedError("cf. https://github.com/pykeen/pykeen/pull/728")
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

    # Prepare for result filtering
    if evaluator.filtered or evaluator.requires_positive_mask:
        all_pos_triples = prepare_filter_triples(
            mapped_triples=mapped_triples,
            additional_filter_triples=additional_filter_triples,
        ).to(device=device)
    else:
        all_pos_triples = None

    # Send tensors to device
    mapped_triples = mapped_triples.to(device=device)

    # Prepare batches
    if batch_size is None:
        # This should be a reasonable default size that works on most setups while being faster than batch_size=1
        batch_size = 32
        logger.info(f"No evaluation batch_size provided. Setting batch_size to '{batch_size}'.")
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
        for batch in mapped_triples.split(split_size=batch_size):
            batch_size = batch.shape[0]
            relation_filter = None
            for target in targets:
                relation_filter = _evaluate_batch(
                    batch=batch,
                    model=model,
                    target=target,
                    evaluator=evaluator,
                    slice_size=slice_size,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                    restrict_entities_to=restrict_entities_to,
                    mode=mode,
                )

            # If we only probe sizes we do not need more than one batch
            if only_size_probing and evaluated_once:
                break

            evaluated_once = True

            if use_tqdm:
                progress_bar.update(batch_size)

        # Finalize
        result = evaluator.finalize()

    stop = timeit.default_timer()
    if only_size_probing:
        logger.debug("Evaluation took %.2fs seconds", stop - start)
    else:
        logger.info("Evaluation took %.2fs seconds", stop - start)

    return result


def _evaluate_batch(
    batch: MappedTriples,
    model: Model,
    target: Target,
    evaluator: Evaluator,
    slice_size: Optional[int],
    all_pos_triples: Optional[MappedTriples],
    relation_filter: Optional[torch.BoolTensor],
    restrict_entities_to: Optional[torch.LongTensor],
    *,
    mode: Optional[InductiveMode],
) -> torch.BoolTensor:
    """
    Evaluate ranking for batch.

    :param batch: shape: (batch_size, 3)
        The batch of currently evaluated triples.
    :param model:
        The model to evaluate.
    :param target:
        The prediction target.
    :param evaluator:
        The evaluator
    :param slice_size:
        An optional slice size for computing the scores.
    :param all_pos_triples:
        All positive triples (required if filtering is necessary).
    :param relation_filter:
        The relation filter. Can be re-used.
    :param restrict_entities_to:
        Restriction to evaluate only for these entities.
    :param mode:
        the inductive mode, or None for transductive evaluation

    :raises ValueError:
        if all positive triples are required (either due to filtered evaluation, or requiring dense masks).

    :return:
        The relation filter, which can be re-used for the same batch.
    """
    scores = model.predict(hrt_batch=batch, target=target, slice_size=slice_size, mode=mode)

    if evaluator.filtered or evaluator.requires_positive_mask:
        column = TARGET_TO_INDEX[target]
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
    else:
        positive_filter = relation_filter = None

    if evaluator.filtered:
        assert positive_filter is not None
        # Select scores of true
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, column]]
        # overwrite filtered scores
        scores = filter_scores_(scores=scores, filter_batch=positive_filter)
        # The scores for the true triples have to be rewritten to the scores tensor
        scores[torch.arange(0, batch.shape[0]), batch[:, column]] = true_scores
        # the rank-based evaluators needs the true scores with trailing 1-dim
        true_scores = true_scores.unsqueeze(dim=-1)
    else:
        true_scores = None

    # Create a positive mask with the size of the scores from the positive filter
    if evaluator.requires_positive_mask:
        assert positive_filter is not None
        positive_mask = create_dense_positive_mask_(zero_tensor=torch.zeros_like(scores), filter_batch=positive_filter)
    else:
        positive_mask = None

    # Restrict to entities of interest
    if restrict_entities_to is not None:
        scores = scores[:, restrict_entities_to]
        if positive_mask is not None:
            positive_mask = positive_mask[:, restrict_entities_to]

    # process scores
    evaluator.process_scores_(
        hrt_batch=batch,
        target=target,
        true_scores=true_scores,
        scores=scores,
        dense_positive_mask=positive_mask,
    )

    return relation_filter


def get_candidate_set_size(
    mapped_triples: MappedTriples,
    restrict_entities_to: Optional[Collection[int]] = None,
    restrict_relations_to: Optional[Collection[int]] = None,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    num_entities: Optional[int] = None,
) -> pandas.DataFrame:
    """
    Calculate the candidate set sizes for head/tail prediction for the given triples.

    :param mapped_triples: shape: (n, 3)
        the evaluation triples
    :param restrict_entities_to:
        The entity IDs of interest. If None, defaults to all entities. cf. :func:`restrict_triples`.
    :param restrict_relations_to:
        The relations IDs of interest. If None, defaults to all relations. cf. :func:`restrict_triples`.
    :param additional_filter_triples: shape: (n, 3)
        additional filter triples besides the evaluation triples themselves. cf. `_prepare_filter_triples`.
    :param num_entities:
        the number of entities. If not given, this number is inferred from all triples

    :return: columns: "index" | "head" | "relation" | "tail" | "head_candidates" | "tail_candidates"
        a dataframe of all evaluation triples, with the number of head and tail candidates
    """
    # optionally restrict triples (nop if no restriction)
    mapped_triples = restrict_triples(
        mapped_triples=mapped_triples,
        entities=restrict_entities_to,
        relations=restrict_relations_to,
    )

    # evaluation triples as dataframe
    df_eval = pandas.DataFrame(
        data=mapped_triples.numpy(),
        columns=COLUMN_LABELS,
    ).reset_index()

    # determine filter triples
    filter_triples = prepare_filter_triples(
        mapped_triples=mapped_triples,
        additional_filter_triples=additional_filter_triples,
    )

    # infer num_entities if not given
    if restrict_entities_to:
        num_entities = len(restrict_entities_to)
    else:
        # TODO: unique, or max ID + 1?
        num_entities = num_entities or filter_triples[:, [0, 2]].view(-1).unique().numel()

    # optionally restrict triples
    filter_triples = restrict_triples(
        mapped_triples=filter_triples,
        entities=restrict_entities_to,
        relations=restrict_relations_to,
    )
    df_filter = pandas.DataFrame(
        data=filter_triples.numpy(),
        columns=COLUMN_LABELS,
    )

    # compute candidate set sizes for different targets
    # TODO: extend to relations?
    for target in [LABEL_HEAD, LABEL_TAIL]:
        total = num_entities
        group_keys = TARGET_TO_KEY_LABELS[target]
        df_count = df_filter.groupby(by=group_keys).agg({target: "count"})
        column = f"{target}_candidates"
        df_count[column] = total - df_count[target]
        df_count = df_count.drop(columns=target)
        df_eval = pandas.merge(df_eval, df_count, on=group_keys, how="left")
        df_eval[column] = df_eval[column].fillna(value=total)

    return df_eval
