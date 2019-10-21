# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

import logging
import timeit
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Collection, List, Optional, Tuple, Union

import torch
from dataclasses_json import dataclass_json
from tqdm import tqdm

from ..models.base import BaseModule
from ..typing import MappedTriples
from ..utils import split_list_in_batches_iter

__all__ = [
    'Evaluator',
    'MetricResults',
    'filter_scores_',
    'evaluate',
]

_LOGGER = logging.getLogger(__name__)


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


class Evaluator(ABC):
    """An abstract evaluator for KGE models.

    The evaluator encapsulates the computation of evaluation metrics based on subject and object scores. To this end, it
    offers two methods to process a batch of triples together with the scores produced by some model. It maintains
    intermediate results in its state, and offers a method to obtain the final results once finished.
    """

    def __init__(
        self,
        filtered: bool = False,
    ):
        self.filtered = filtered

    @abstractmethod
    def process_object_scores_(
        self,
        batch: MappedTriples,
        scores: torch.FloatTensor,
    ) -> None:
        """Process a batch of triples with their computed object scores for all entities.

        :param batch: shape: (batch_size, 3)
        :param scores: shape: (batch_size, num_entities)
        """
        raise NotImplementedError

    @abstractmethod
    def process_subject_scores_(
        self,
        batch: MappedTriples,
        scores: torch.FloatTensor,
    ) -> None:
        """Process a batch of triples with their computed subject scores for all entities.

        :param batch: shape: (batch_size, 3)
        :param scores: shape: (batch_size, num_entities)
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> MetricResults:
        """Compute the final results, and clear buffers."""
        raise NotImplementedError

    def evaluate(
        self,
        model: BaseModule,
        mapped_triples: Optional[MappedTriples] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        use_tqdm: bool = True,
    ) -> MetricResults:
        """Run :func:`poem.evaluation.evaluate` with this evaluator."""
        if mapped_triples is None:
            mapped_triples = model.triples_factory.mapped_triples
        return evaluate(
            model=model,
            mapped_triples=mapped_triples,
            evaluators=self,
            batch_size=batch_size,
            device=device,
            squeeze=True,
            use_tqdm=use_tqdm,
        )


def filter_scores_(
    batch: MappedTriples,
    scores: torch.FloatTensor,
    all_pos_triples: torch.LongTensor,
    relation_filter: torch.BoolTensor = None,
    filter_col: int = 0,
) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    """Filter scores.

    For simplicity, only the subject-side is described, i.e. filter_col=0. The object-side is processed alike.

    For each (s, p, o) triple in the batch, the triple scores for (s', p, o) are set to -infinity if the positive triple
    (s',p,o) exists in all positive triples. Thereby, subjects other than the current subject that are already contained
    in the all_pos_triples tensor as true are not considered and thus, do not penalise the rank of the currently
    considered subject.

    :param batch: shape: (batch_size, 3)
        A batch of triples.
    :param scores: shape: (batch_size, num_entities)
        The scores for all corrupted triples (including the currently considered true triple). Are modified *in-place*.
    :param all_pos_triples: shape: (num_positive_triples, 3)
        All positive triples to base the filtering on.
    :param all_entities: shape: (num_entities,) (optional)
        A tensor containing all entity IDs. Should equal torch.arange(num_entities), and may be passed to avoid numerous
        re-constructions of the same tensor.
    :param relation_filter: shape: (batch_size, num_positive_triples)
        A boolean mask R[i, j] which is True iff the j-th positive triple contains the same relation as the i-th triple
        in the batch.
    :param filter_col:
        The column along which to filter. Allowed are {0, 2}, where 0 corresponds to filtering subject-based and 2
        corresponds to filtering object-based.

    :return:
        - A reference to the scores, which have been updated in-place.
        - the relation filter for re-usage.
    """
    if filter_col not in {0, 2}:
        raise NotImplementedError(
            'This code has only been written for updating subject (filter_col=0) or '
            f'object (filter_col=1) mask, but filter_col={filter_col} was given.',
        )

    # Bind shape
    batch_size, num_entities = scores.shape

    if relation_filter is None:
        relations = batch[:, 1:2]
        relation_filter = (all_pos_triples[:, 1:2]).view(1, -1) == relations

    # Split batch
    other_col = 2 - filter_col
    entities = batch[:, other_col:other_col + 1]

    entity_filter_test = (all_pos_triples[:, other_col:other_col + 1]).view(1, -1) == entities
    filter_batch = (entity_filter_test & relation_filter).nonzero()
    filter_batch[:, 1] = all_pos_triples[:, filter_col:filter_col + 1].view(1, -1)[:, filter_batch[:, 1]]

    # Set all filtered triples to NaN to ensure their exclusion in subsequent calculations
    scores[filter_batch[:, 0], filter_batch[:, 1]] = float('nan')

    # Warn if all entities will be filtered
    # (scores != scores) yields true for all NaN instances (IEEE 754), thus allowing to count the filtered triples.
    if ((scores != scores).sum(dim=1) == num_entities).any():
        _LOGGER.warning(
            "User selected filtered metric computation, but all corrupted triples exists also as positive "
            "triples",
        )

    return scores, relation_filter


def evaluate(
    model: BaseModule,
    mapped_triples: MappedTriples,
    evaluators: Union[Evaluator, Collection[Evaluator]],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    squeeze: bool = True,
    use_tqdm: bool = True,
) -> Union[MetricResults, List[MetricResults]]:
    """Evaluate metrics for model on mapped triples.

    The model is used to predict scores for all objects and all subjects for each triple. Subsequently, each abstract
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
    :param batch_size: >0
        A positive integer used as batch size. Generally chosen as large as possible.
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

    # Prepare for result filtering
    if filtering_necessary:
        all_pos_triples = torch.cat([model.triples_factory.mapped_triples, mapped_triples], dim=0)
        all_pos_triples = all_pos_triples.to(device=device)
    else:
        all_pos_triples = None

    # Send tensors to device
    mapped_triples = mapped_triples.to(device=device)

    # Prepare batches
    batches = split_list_in_batches_iter(input_list=mapped_triples, batch_size=batch_size)

    # Show progressbar
    num_triples = mapped_triples.shape[0]

    # Disable gradient tracking
    with optional_context_manager(
        use_tqdm,
        tqdm(
            desc=f'Evaluating on {model.device}',
            total=num_triples,
            unit='triple(s)',
            unit_scale=True,
        ),
    ) as progress_bar, torch.no_grad():
        # batch-wise processing
        for batch in batches:
            batch_size = batch.shape[0]

            # Predict object scores once
            scores_of_corrupted_objects_batch = model.predict_scores_all_objects(batch[:, 0:2])

            # Evaluate metrics on these *unfiltered* object scores
            for unfiltered_evaluator in unfiltered_evaluators:
                unfiltered_evaluator.process_object_scores_(
                    batch=batch,
                    scores=scores_of_corrupted_objects_batch,
                )

            # Filter
            if filtering_necessary:
                assert all_pos_triples is not None
                filtered_scores_of_corrupted_objects_batch, relation_filter = filter_scores_(
                    batch=batch,
                    scores=scores_of_corrupted_objects_batch,
                    all_pos_triples=all_pos_triples,
                    relation_filter=None,
                    filter_col=2,
                )

                # Evaluate metrics on these *filtered* object scores
                for filtered_evaluator in filtered_evaluators:
                    filtered_evaluator.process_object_scores_(
                        batch=batch,
                        scores=filtered_scores_of_corrupted_objects_batch,
                    )

            # Predict subject scores once
            scores_of_corrupted_subjects_batch = model.predict_scores_all_subjects(batch[:, 1:3])

            # Evaluate metrics on these subject scores
            for evaluator in unfiltered_evaluators:
                evaluator.process_subject_scores_(
                    batch=batch,
                    scores=scores_of_corrupted_subjects_batch,
                )

            # Filter
            if filtering_necessary:
                assert all_pos_triples is not None
                assert relation_filter is not None
                filtered_scores_of_corrupted_subjects_batch, _ = filter_scores_(
                    batch=batch,
                    scores=scores_of_corrupted_subjects_batch,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                    filter_col=0,
                )

                # Evaluate metrics on these *filtered* object scores
                for filtered_evaluator in filtered_evaluators:
                    filtered_evaluator.process_object_scores_(
                        batch=batch,
                        scores=filtered_scores_of_corrupted_subjects_batch,
                    )

            if use_tqdm:
                progress_bar.update(batch_size)

        # Finalize
        results = [evaluator.finalize() for evaluator in evaluators]

    stop = timeit.default_timer()
    _LOGGER.info("Evaluation took %.2fs seconds", stop - start)

    if squeeze and len(results) == 1:
        return results[0]

    return results
