# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

import gc
import logging
import timeit
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Any, Collection, List, Mapping, Optional, Tuple, Union

import torch
from dataclasses_json import dataclass_json
from tqdm import tqdm

from ..models.base import Model
from ..typing import MappedTriples
from ..utils import is_cuda_oom_error, is_cudnn_error, normalize_string, split_list_in_batches_iter

__all__ = [
    'Evaluator',
    'MetricResults',
    'filter_scores_',
    'evaluate',
]

logger = logging.getLogger(__name__)


@contextmanager
def optional_context_manager(condition, context_manager):
    if condition:
        with context_manager:
            yield context_manager
    else:
        yield


@dataclass_json
@dataclass
class MetricResults:
    """Results from computing metrics."""

    def get_metric(self, name: str) -> float:
        """Get the given metric from the results."""
        raise NotImplementedError

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
        batch_size: int = None,
        slice_size: int = None,
    ):
        self.filtered = filtered
        self.requires_positive_mask = requires_positive_mask
        self.batch_size = batch_size
        self.slice_size = slice_size

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the evaluator."""
        return normalize_string(cls.__name__, suffix=Evaluator.__name__)

    @abstractmethod
    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
        """Process a batch of triples with their computed tail scores for all entities.

        :param hrt_batch: shape: (batch_size, 3)
        :param true_scores: shape: (batch_size)
        :param scores: shape: (batch_size, num_entities)
        :param dense_positive_mask: shape: (batch_size, num_entities)
            An optional binary (0/1) tensor indicating other true entities.
        """
        raise NotImplementedError

    @abstractmethod
    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
        """Process a batch of triples with their computed head scores for all entities.

        :param hrt_batch: shape: (batch_size, 3)
        :param true_scores: shape: (batch_size)
        :param scores: shape: (batch_size, num_entities)
        :param dense_positive_mask: shape: (batch_size, num_entities)
            An optional binary (0/1) tensor indicating other true entities.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> MetricResults:
        """Compute the final results, and clear buffers."""
        raise NotImplementedError

    def evaluate(
        self,
        model: Model,
        mapped_triples: Optional[MappedTriples] = None,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_tqdm: bool = True,
    ) -> MetricResults:
        """Run :func:`pykeen.evaluation.evaluate` with this evaluator."""
        if mapped_triples is None:
            mapped_triples = model.triples_factory.mapped_triples

        if batch_size is None and model.automatic_memory_optimization:
            batch_size, slice_size = self.batch_and_slice(
                model=model,
                mapped_triples=mapped_triples,
                batch_size=batch_size,
                device=device,
                use_tqdm=False,
            )
            # The batch_size and slice_size should be accessible to outside objects for re-use, e.g. early stoppers.
            self.batch_size = batch_size
            self.slice_size = slice_size

            # Clear the ranks from the current evaluator
            self.finalize()

        return evaluate(
            model=model,
            mapped_triples=mapped_triples,
            evaluators=self,
            batch_size=batch_size,
            slice_size=slice_size,
            device=device,
            squeeze=True,
            use_tqdm=use_tqdm,
        )

    def batch_and_slice(
        self,
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_tqdm: bool = False,
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
        :param device:
            The device on which the evaluation shall be run. If None is given, use the model's device.
        :param use_tqdm:
            Should a progress bar be displayed?

        :return:
            Maximum possible batch size and, if necessary, the slice_size, which defaults to None.

        :raises MemoryError:
            If it is not possible to evaluate the model on the hardware at hand with the given parameters.
        """
        batch_size, evaluated_once = self._param_size_search(
            key='batch_size',
            start_value=batch_size,
            model=model,
            mapped_triples=mapped_triples,
            device=device,
            use_tqdm=use_tqdm,
        )

        if evaluated_once:  # slice_size = None
            return batch_size, None

        # We need to try slicing, if the evaluation for the batch_size search never succeeded
        slice_size, evaluated_once = self._param_size_search(
            key='slice_size',
            # Since the batch_size search with size 1, i.e. one tuple ((h, r) or (r, t)) scored on all entities,
            # must have failed to start slice_size search, we start with trying half the entities.
            start_value=ceil(model.num_entities / 2),
            model=model,
            mapped_triples=mapped_triples,
            device=device,
            use_tqdm=use_tqdm,
        )
        if not evaluated_once:
            raise MemoryError("The current model can't be trained on this hardware with these parameters.")

        return batch_size, slice_size

    def _param_size_search(
        self,
        key: str,
        start_value: int,
        model: Model,
        mapped_triples: MappedTriples,
        device: Optional[torch.device] = None,
        use_tqdm: bool = False,
    ) -> Tuple[int, bool]:
        values_dict = {}
        maximum_triples = mapped_triples.shape[0]
        if key == 'batch_size':
            if start_value is None:
                start_value = 256
            if start_value > maximum_triples:
                start_value = maximum_triples
            values_dict[key] = start_value
            values_dict['slice_size'] = None
        elif key == 'slice_size':
            self._check_slicing_availability(model, batch_size=1)
            values_dict[key] = start_value
            values_dict['batch_size'] = 1
        else:
            raise AttributeError(f'The parameter {key} is unknown.')
        reached_max = False
        evaluated_once = False
        logger.info(f'Starting {key} search for evaluation now...')
        while True:
            logger.debug(f'Trying {key}={values_dict[key]}')
            try:
                # The cache of the previous run has to be freed to allow accurate memory availability estimates
                gc.collect()
                torch.cuda.empty_cache()
                evaluate(
                    **values_dict,
                    model=model,
                    mapped_triples=mapped_triples,
                    evaluators=self,
                    only_size_probing=True,
                    device=device,
                    squeeze=True,
                    use_tqdm=use_tqdm,
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
                if not is_cudnn_error(runtime_error) and not is_cuda_oom_error(runtime_error):
                    raise runtime_error
                if values_dict[key] == 1:
                    logger.debug(
                        f"Even {key} {values_dict[key]} does not fit into your memory with these parameters."
                    )
                    break

                values_dict[key] //= 2
                reached_max = True
                if evaluated_once:
                    logger.info(f'Concluded {key} search with batch_size={values_dict[key]}.')
                    break
                else:
                    logger.debug(f'The {key} {values_dict[key]} was too big, trying less now')
            else:
                # The cache of the previous run has to be freed to allow accurate memory availability estimates
                gc.collect()
                torch.cuda.empty_cache()
                if not reached_max and values_dict['batch_size'] < maximum_triples:
                    values_dict[key] *= 2
                else:
                    logger.info(f'Concluded {key} search with batch_size={values_dict[key]}.')
                    break

        return values_dict[key], evaluated_once

    @staticmethod
    def _check_slicing_availability(model: Model, batch_size: int) -> None:
        # Test if slicing is implemented for the required functions of this model
        if model.triples_factory.create_inverse_triples:
            if not model.can_slice_t:
                raise MemoryError(f"The current model can't be evaluated on this hardware with these parameters, as "
                                  f"evaluation batch_size={batch_size} is too big and slicing is not implemented for "
                                  f"this model yet.")
        elif not model.can_slice_t or not model.can_slice_h:
            raise MemoryError(f"The current model can't be evaluated on this hardware with these parameters, as "
                              f"evaluation batch_size={batch_size} is too big and slicing is not implemented for this "
                              f"model yet.")


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
    """
    if filter_col not in {0, 2}:
        raise NotImplementedError(
            'This code has only been written for updating head (filter_col=0) or '
            f'tail (filter_col=2) mask, but filter_col={filter_col} was given.',
        )

    if relation_filter is None:
        relations = hrt_batch[:, 1:2]
        relation_filter = (all_pos_triples[:, 1:2]).view(1, -1) == relations

    # Split batch
    other_col = 2 - filter_col
    entities = hrt_batch[:, other_col:other_col + 1]

    entity_filter_test = (all_pos_triples[:, other_col:other_col + 1]).view(1, -1) == entities
    filter_batch = (entity_filter_test & relation_filter).nonzero()
    filter_batch[:, 1] = all_pos_triples[:, filter_col:filter_col + 1].view(1, -1)[:, filter_batch[:, 1]]

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
    scores[filter_batch[:, 0], filter_batch[:, 1]] = float('nan')

    # Warn if all entities will be filtered
    # (scores != scores) yields true for all NaN instances (IEEE 754), thus allowing to count the filtered triples.
    if ((scores != scores).sum(dim=1) == num_entities).any():
        logger.warning(
            "User selected filtered metric computation, but all corrupted triples exists also as positive "
            "triples",
        )

    return scores


def evaluate(
    model: Model,
    mapped_triples: MappedTriples,
    evaluators: Union[Evaluator, Collection[Evaluator]],
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    slice_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    squeeze: bool = True,
    use_tqdm: bool = True,
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
        The triples on which to evaluate.
    :param evaluators:
        An evaluator or a list of evaluators working on batches of triples and corresponding scores.
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
    """
    if isinstance(evaluators, Evaluator):  # upgrade a single evaluator to a list
        evaluators = [evaluators]

    start = timeit.default_timer()

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
        all_pos_triples = torch.cat([model.triples_factory.mapped_triples, mapped_triples], dim=0)
        all_pos_triples = all_pos_triples.to(device=device)
    else:
        all_pos_triples = None

    # Send tensors to device
    mapped_triples = mapped_triples.to(device=device)

    # Prepare batches
    if batch_size is None:
        batch_size = 1
    batches = split_list_in_batches_iter(input_list=mapped_triples, batch_size=batch_size)

    # Show progressbar
    num_triples = mapped_triples.shape[0]

    # Flag to check when to quit the size probing
    evaluated_once = False

    # Disable gradient tracking
    with optional_context_manager(
        use_tqdm,
        tqdm(
            desc=f'Evaluating on {model.device}',
            total=num_triples,
            unit='triple',
            unit_scale=True,
            # Choosing no progress bar (use_tqdm=False) would still show the initial progress bar without disable=True
            disable=not use_tqdm,
        ),
    ) as progress_bar, torch.no_grad():
        # batch-wise processing
        for batch in batches:
            batch_size = batch.shape[0]

            # Predict tail scores once
            scores_of_corrupted_tails_batch = model.predict_scores_all_tails(batch[:, 0:2], slice_size=slice_size)
            scores_of_true_tails_batch = scores_of_corrupted_tails_batch[
                torch.arange(0, batch.shape[0]),
                batch[:, 2],
            ]

            # Create positive filter for all corrupted tails
            if filtering_necessary or positive_masks_required:
                assert all_pos_triples is not None
                positive_filter_tails, relation_filter = create_sparse_positive_filter_(
                    hrt_batch=batch,
                    all_pos_triples=all_pos_triples,
                    relation_filter=None,
                    filter_col=2,
                )

            # Create a positive mask with the size of the scores from the positive tails filter
            if positive_masks_required:
                positive_mask_tails = create_dense_positive_mask_(
                    zero_tensor=torch.zeros_like(scores_of_corrupted_tails_batch),
                    filter_batch=positive_filter_tails,
                )
            else:
                positive_mask_tails = None

            # Evaluate metrics on these *unfiltered* tail scores
            for unfiltered_evaluator in unfiltered_evaluators:
                unfiltered_evaluator.process_tail_scores_(
                    hrt_batch=batch,
                    true_scores=scores_of_true_tails_batch[:, None],
                    scores=scores_of_corrupted_tails_batch,
                    dense_positive_mask=positive_mask_tails,
                )

            # Filter
            if filtering_necessary:
                filtered_scores_of_corrupted_tails_batch = filter_scores_(
                    scores=scores_of_corrupted_tails_batch,
                    filter_batch=positive_filter_tails,
                )

                # The scores for the true triples have to be rewritten to the scores tensor
                scores_of_corrupted_tails_batch[
                    torch.arange(0, batch.shape[0]),
                    batch[:, 2],
                ] = scores_of_true_tails_batch

                # Evaluate metrics on these *filtered* tail scores
                for filtered_evaluator in filtered_evaluators:
                    filtered_evaluator.process_tail_scores_(
                        hrt_batch=batch,
                        true_scores=scores_of_true_tails_batch[:, None],
                        scores=filtered_scores_of_corrupted_tails_batch,
                    )

            # Predict head scores once
            scores_of_corrupted_heads_batch = model.predict_scores_all_heads(batch[:, 1:3], slice_size=slice_size)
            scores_of_true_heads_batch = scores_of_corrupted_heads_batch[
                torch.arange(0, batch.shape[0]),
                batch[:, 0],
            ]

            # Create positive filter for all corrupted heads
            if filtering_necessary or positive_masks_required:
                assert all_pos_triples is not None
                assert relation_filter is not None
                positive_filter_heads, _ = create_sparse_positive_filter_(
                    hrt_batch=batch,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                    filter_col=0,
                )

            # Create a positive mask with the size of the scores from the positive heads filter
            if positive_masks_required:
                positive_mask_heads = create_dense_positive_mask_(
                    zero_tensor=torch.zeros_like(scores_of_corrupted_heads_batch),
                    filter_batch=positive_filter_heads,
                )
            else:
                positive_mask_heads = None

            # Evaluate metrics on these head scores
            for evaluator in unfiltered_evaluators:
                evaluator.process_head_scores_(
                    hrt_batch=batch,
                    true_scores=scores_of_true_heads_batch[:, None],
                    scores=scores_of_corrupted_heads_batch,
                    dense_positive_mask=positive_mask_heads,
                )

            # Filter
            if filtering_necessary:
                assert all_pos_triples is not None
                assert relation_filter is not None
                filtered_scores_of_corrupted_heads_batch = filter_scores_(
                    scores=scores_of_corrupted_heads_batch,
                    filter_batch=positive_filter_heads
                )

                # The scores for the true triples have to be rewritten to the scores tensor
                scores_of_corrupted_heads_batch[
                    torch.arange(0, batch.shape[0]),
                    batch[:, 0],
                ] = scores_of_true_heads_batch

                # Evaluate metrics on these *filtered* tail scores
                for filtered_evaluator in filtered_evaluators:
                    filtered_evaluator.process_head_scores_(
                        hrt_batch=batch,
                        true_scores=scores_of_true_heads_batch[:, None],
                        scores=filtered_scores_of_corrupted_heads_batch,
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
