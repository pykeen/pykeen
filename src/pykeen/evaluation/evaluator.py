# -*- coding: utf-8 -*-

"""Basic structure of a evaluator."""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, ClassVar, Collection, List, Mapping, Optional, Type, Union

import pandas
import torch

from .evaluation_loop import LCWAEvaluationLoop
from ..constants import COLUMN_LABELS, TARGET_TO_KEY_LABELS
from ..metrics.utils import Metric
from ..models import Model
from ..triples.triples_factory import restrict_triples
from ..typing import LABEL_HEAD, LABEL_TAIL, InductiveMode, MappedTriples, Target
from ..utils import normalize_string, prepare_filter_triples

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
    def finalize(self) -> MetricResults:
        """Compute the final results, and clear buffers."""
        raise NotImplementedError

    def evaluate(
        self,
        model: Model,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        use_tqdm: bool = True,
        tqdm_kwargs: Optional[Mapping[str, str]] = None,
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
        loop = LCWAEvaluationLoop(model=model, **kwargs)
        return loop.evaluate(batch_size=batch_size, use_tqdm=use_tqdm, tqdm_kwargs=tqdm_kwargs)

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
