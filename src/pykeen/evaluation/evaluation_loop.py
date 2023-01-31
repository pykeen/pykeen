# -*- coding: utf-8 -*-

"""Evaluation loops for KGE models."""

import dataclasses
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Collection, DefaultDict, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union, cast

import numpy
import pandas
import torch
from class_resolver import HintOrType, OptionalKwargs
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm
from typing_extensions import TypeAlias

from .evaluator import Evaluator, MetricResults, filter_scores_
from ..constants import COLUMN_LABELS, TARGET_TO_INDEX
from ..models import Model
from ..triples import CoreTriplesFactory, get_mapped_triples
from ..typing import LABEL_HEAD, LABEL_TAIL, InductiveMode, MappedTriples, OneOrSequence, Target
from ..utils import upgrade_to_sequence

__all__ = [
    "AdditionalFilterTriplesHint",
    # Evaluation loops
    "EvaluationLoop",
    "LCWAEvaluationLoop",
    # Evaluation datasets
    "LCWAEvaluationDataset",
]

logger = logging.getLogger(__name__)

BatchType = TypeVar("BatchType")
AdditionalFilterTriplesHint: TypeAlias = Optional[OneOrSequence[Union[MappedTriples, CoreTriplesFactory]]]


def _hasher(d: Mapping[str, Any]) -> int:
    """
    Calculate hash based on ID of dataset.

    This means that we can have separate batch sizes for different evaluation datasets.

    :param d:
        the dictionary of keyword-based parameters

    :return:
        the dataset's ID
    """
    obj = d["loop"]
    assert isinstance(obj, EvaluationLoop)
    obj = obj.dataset
    return id(obj)


#: the MemoryUtilizationMaximizer instance for :func:`_evaluate`.
evaluation_batch_size_maximizer = MemoryUtilizationMaximizer(hasher=_hasher)


@evaluation_batch_size_maximizer
def _evaluate(
    loop: "EvaluationLoop",
    batch_size: int,
    use_tqdm: bool,
    tqdm_kwargs: OptionalKwargs,
    **kwargs,
) -> MetricResults:
    """
    Run the evaluation loop for a given batch size.

    .. note::
        this method is wrapped into a `MemoryUtilizationMaximizer` instance to automatically tune the `batch_size`.

    :param loop:
        the evaluation loop instance.
    :param batch_size:
        the batch size
    :param use_tqdm:
        whether to use tqdm progress bar
    :param tqdm_kwargs:
        additional keyword-based parameters for the progress bar
    :param kwargs:
        additional keyword-based parameters passed to :meth:`EvaluationLoop.get_loader`

    :return:
        the evaluation results
    """
    loop.model.eval()
    loader = loop.get_loader(batch_size=batch_size, **kwargs)
    total = len(loader)
    if use_tqdm:
        loader = tqdm(
            loader,
            desc="evaluation",
            total=total,
            unit="batch",
            unit_scale=True,
            **(tqdm_kwargs or {}),
        )
    for batch in loader:
        loop.process_batch(batch=batch)
    return loop.evaluator.finalize()


class EvaluationLoop(Generic[BatchType]):
    """A base class for evaluation loops."""

    def __init__(
        self,
        model: Model,
        dataset: Dataset[BatchType],
        evaluator: Evaluator,
    ) -> None:
        """
        Initialize the evaluation loop.

        :param model:
            the model to evaluate.
        :param dataset:
            the evaluation dataset
        :param evaluator:
            the evaluator instance
        """
        self.model = model
        self.evaluator = evaluator
        self.dataset = dataset

    @abstractmethod
    def process_batch(self, batch: BatchType) -> None:
        """
        Process a single batch.

        :param batch:
            one batch of evaluation samples from the dataset.
        """
        raise NotImplementedError

    def get_collator(self):
        """Get the collator to use for the data loader."""
        return None

    def get_loader(self, batch_size: int, pin_memory: bool = True, **kwargs) -> DataLoader:
        """
        Create a data loader for a single evaluation round.

        :param batch_size:
            the batch size
        :param pin_memory:
            whether to pin memory, cf. :meth:`DataLoader.__init__`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`DataLoader.__init__`

        :return:
            a dataloader for the evaluation dataset of the given batch size
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=self.get_collator(),
            **kwargs,
        )

    @torch.inference_mode()
    def evaluate(
        self,
        # batch
        batch_size: Optional[int] = None,
        # tqdm
        use_tqdm: bool = True,
        tqdm_kwargs: OptionalKwargs = None,
        # data loader
        **kwargs,
    ) -> MetricResults:
        """
        Evaluate the loop's model on the loop's dataset.

        .. note::
            the contained model will be set to evaluation mode.

        :param batch_size:
            the batch size. If None, enable automatic memory optimization to maximize memory utilization.
        :param use_tqdm:
            whether to use tqdm progress bar
        :param tqdm_kwargs:
            additional keyword-based parameters passed to tqdm
        :param kwargs:
            additional keyword-based parameters passed to :meth:`get_loader`

        :return:
            the evaluation results.
        """
        # set upper limit of batch size for automatic memory optimization
        if not batch_size:
            if self.model.device.type == "cpu":
                batch_size = 32
            else:
                batch_size = len(self.dataset)
        # set model to evaluation mode
        self.model.eval()
        # delegate to AMO wrapper
        return _evaluate(
            loop=self,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
            tqdm_kwargs=tqdm_kwargs,
            **kwargs,
        )


@dataclasses.dataclass
class FilterIndex:
    """An index structure for filtering (roughly following CSR)."""

    # The key-id for each triple, shape: (num_triples,)
    triple_id_to_key_id: numpy.ndarray

    #: the number of targets for each key, shape: (num_unique_keys + 1,)
    bounds: numpy.ndarray

    #: the concatenation of unique targets for each key (use bounds to select appropriate sub-array)
    indices: torch.LongTensor

    @classmethod
    def from_df(cls, df: pandas.DataFrame, target: Target) -> "FilterIndex":
        """
        Create index from dataframe.

        :param df:
            the dataframe, comprising columns [LABEL_HEAD, LABEL_RELATION, LABEL_TAIL]
        :param target:
            the prediction target

        :raises ValueError:
            if some of the expected columns are missing

        :return:
            a filter index object
        """
        # input verification
        expected_columns = set(COLUMN_LABELS)
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"Missing columns: {sorted(expected_columns.difference(df.columns))}")

        # group key = everything except the prediction target
        key = [c for c in df.columns if c != target]
        # initialize data structure
        triple_id_to_key_id = numpy.empty_like(df.index)
        indices = []
        bounds = [0]
        # group by key
        for key_id, (_, group) in enumerate(df.groupby(by=key)):
            unique_targets = group[target].unique()
            triple_id_to_key_id[group.index] = key_id
            indices.extend(unique_targets)
            bounds.append(len(indices))
        # convert lists to arrays
        indices = cast(torch.LongTensor, torch.as_tensor(indices))
        bounds = numpy.asarray(bounds)
        # instantiate
        return cls(triple_id_to_key_id=triple_id_to_key_id, bounds=bounds, indices=indices)

    def __getitem__(self, item: int) -> numpy.ndarray:  # noqa: D105
        # return indices corresponding to the `item`-th triple
        key_id = self.triple_id_to_key_id[item]
        low, high = self.bounds[key_id : key_id + 2]
        return self.indices[low:high]


class LCWAEvaluationDataset(Dataset[Mapping[Target, Tuple[MappedTriples, Optional[torch.Tensor]]]]):
    """A dataset for link prediction evaluation."""

    filter_indices: Optional[Mapping[Target, FilterIndex]]

    def __init__(
        self,
        *,
        mapped_triples: Optional[MappedTriples] = None,
        factory: Optional[CoreTriplesFactory] = None,
        targets: Optional[Collection[Target]] = None,
        filtered: bool = True,
        additional_filter_triples: AdditionalFilterTriplesHint = None,
    ) -> None:
        """
        Create a PyTorch dataset for link prediction evaluation.

        :param mapped_triples: shape: (n, 3)
            the ID-based triples
        :param factory:
            the triples factory. Only used of `mapped_triples` is None
        :param targets:
            the prediction targets. Defaults to head and tail prediction
        :param filtered:
            whether to use filtered evaluation, i.e., prepare filter indices
        :param additional_filter_triples:
            additional filter triples to use for creating the filter
        """
        super().__init__()

        # input normalization
        if targets is None:
            targets = [LABEL_HEAD, LABEL_TAIL]
        mapped_triples = get_mapped_triples(mapped_triples=mapped_triples, factory=factory)

        self.mapped_triples = mapped_triples
        self.num_triples = mapped_triples.shape[0]
        self.targets = tuple(targets)

        # prepare filter indices if required
        if filtered:
            if not additional_filter_triples:
                logger.warning("Enabled filtered evaluation, but not additional filter triples are passed.")
            df = pandas.DataFrame(
                data=torch.cat(
                    [
                        mapped_triples,
                        *(get_mapped_triples(x) for x in upgrade_to_sequence(additional_filter_triples or [])),
                    ]
                ),
                columns=COLUMN_LABELS,
            )
            self.filter_indices = {target: FilterIndex.from_df(df=df, target=target) for target in targets}
        else:
            if additional_filter_triples:
                logger.warning("Passed additional filter triples, but filtered evaluation is disabled.")
            self.filter_indices = None

    @property
    def num_targets(self) -> int:
        """Return the number of targets."""
        return len(self.targets)

    def __len__(self) -> int:  # noqa: D105
        return self.num_triples * self.num_targets

    def __getitem__(self, index: int) -> Tuple[Target, MappedTriples, Optional[torch.LongTensor]]:  # noqa: D105
        # sorted by target -> most of the batches only have a single target
        target_id, index = divmod(index, self.num_triples)
        target = self.targets[target_id]
        triple = self.mapped_triples[index, :]
        nnz = None if self.filter_indices is None else self.filter_indices[target][index]
        return target, triple, nnz

    @staticmethod
    def collate(
        batch: Iterable[Tuple[Target, MappedTriples, Optional[torch.LongTensor]]]
    ) -> Mapping[Target, Tuple[MappedTriples, Optional[torch.Tensor]]]:
        """Collate batches by grouping by target."""
        # group by target
        triples: DefaultDict[Target, List[torch.LongTensor]] = defaultdict(list)
        nnz: DefaultDict[Target, List[torch.LongTensor]] = defaultdict(list)
        for target, triple, opt_nnz in batch:
            triples[target].append(triple)
            if opt_nnz is not None:
                nnz[target].append(opt_nnz)

        # stack groups into a single tensor
        result = {}
        for target in triples.keys():
            target_triples = cast(MappedTriples, torch.stack(triples[target]))
            if target in nnz:
                batch_ids = []
                target_nnz = nnz[target]
                for batch_id, size in enumerate(map(len, target_nnz)):
                    batch_ids.append(torch.full(size=(size,), fill_value=batch_id, dtype=torch.long))
                batch_ids = torch.cat(batch_ids)
                target_nnz = torch.cat(target_nnz)
                sparse_filter_mask = torch.stack([batch_ids, target_nnz], dim=-1)
            else:
                sparse_filter_mask = None
            result[target] = (target_triples, sparse_filter_mask)

        return result


class LCWAEvaluationLoop(EvaluationLoop[Mapping[Target, MappedTriples]]):
    r"""
    Evaluation loop using 1:n scoring.

    For brevity, we only describe evaluation for tail prediction. Let $(h, r, t) \in \mathcal{T}_{eval}$ denote an
    evaluation triple. Then, we calculate scores for all triples $(h, r, t')$ with $t' \in \mathcal{E}$, i.e., for
    replacing the true tail $t$ by all entities.
    """

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        evaluator: HintOrType[Evaluator] = None,
        evaluator_kwargs: OptionalKwargs = None,
        targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
        mode: Optional[InductiveMode] = None,
        additional_filter_triples: AdditionalFilterTriplesHint = None,
        **kwargs,
    ) -> None:
        """
        Initialize the evaluation loop.

        :param triples_factory:
            the evaluation triples factory
        :param evaluator:
            the evaluator, or a hint thereof
        :param evaluator_kwargs:
            additional keyword-based parameters for instantiating the evaluator
        :param targets:
            the prediction targets.
        :param mode:
            the inductive mode, or None for transductive evaluation
        :param additional_filter_triples:
            additional filter triples to use for creating the filter
        :param kwargs:
            additional keyword-based parameters passed to :meth:`EvaluationLoop.__init__`. Should not contain the keys
            `dataset` or `evaluator`.
        """
        # avoid cyclic imports
        from . import evaluator_resolver

        # TODO: it would be better to allow separate batch sizes for entity/relation prediction
        evaluator = evaluator_resolver.make(evaluator, pos_kwargs=evaluator_kwargs)
        super().__init__(
            dataset=LCWAEvaluationDataset(
                factory=triples_factory,
                targets=targets,
                filtered=evaluator.filtered or evaluator.requires_positive_mask,
                additional_filter_triples=additional_filter_triples,
            ),
            evaluator=evaluator,
            **kwargs,
        )
        self.targets = targets
        self.mode = mode

    # docstr-coverage: inherited
    def get_collator(self):  # noqa: D102
        return LCWAEvaluationDataset.collate

    # docstr-coverage: inherited
    def process_batch(self, batch: Mapping[Target, MappedTriples]) -> None:  # noqa: D102
        # note: most of the time, this loop will only make a single iteration, since the evaluation dataset typically is
        #       not shuffled, and contains evaluation ranking tasks sorted by target
        for target, (hrt_batch, filter_batch) in batch.items():
            # TODO: in theory, we could make a single score calculation for e.g.,
            # {(h, r, t1), (h, r, t1), ..., (h, r, tk)}
            # predict scores for all candidates
            scores = self.model.predict(hrt_batch=hrt_batch, target=target, mode=self.mode)
            true_scores = dense_positive_mask = None

            # filter scores
            if self.evaluator.filtered:
                if filter_batch is None:
                    raise AssertionError("Filter indices are required to filter scores.")
                # extract true scores
                batch_ids = torch.arange(scores.shape[0], device=scores.device)
                target_ids = hrt_batch[:, TARGET_TO_INDEX[target]]
                true_scores = scores[batch_ids, target_ids, None]
                # replace by nan
                scores = filter_scores_(scores=scores, filter_batch=filter_batch)
                # rewrite true scores
                scores[batch_ids, target_ids] = true_scores[:, 0]

            # create dense positive masks
            # TODO: afaik, dense positive masks are not used on GPU -> we do not need to move the masks around
            elif self.evaluator.requires_positive_mask:
                if filter_batch is None:
                    raise AssertionError("Filter indices are required to create dense positive masks.")
                dense_positive_mask = torch.zeros_like(scores, dtype=torch.bool, device=filter_batch.device)
                dense_positive_mask[filter_batch[:, 0], filter_batch[:, 0]] = True

            # delegate processing of scores to the evaluator
            self.evaluator.process_scores_(
                hrt_batch=hrt_batch,
                target=target,
                scores=scores,
                true_scores=true_scores,
                dense_positive_mask=dense_positive_mask,
            )
