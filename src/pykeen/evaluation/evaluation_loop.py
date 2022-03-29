# -*- coding: utf-8 -*-

"""Evaluation loops for KGE models."""
import dataclasses
import itertools
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Collection, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar, cast

import numpy
import pandas
import torch
from class_resolver import HintOrType, OptionalKwargs
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm

from . import evaluator_resolver
from .evaluator import Evaluator, MetricResults, filter_scores_
from ..constants import TARGET_TO_INDEX
from ..models import Model
from ..triples import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, InductiveMode, MappedTriples, Target

BatchType = TypeVar("BatchType")


def _hasher(d: Mapping[str, Any]) -> int:
    """
    Calculate hash based on ID of dataset.

    This means that we can have separate batch sizes for different evaluation datasets.
    """
    obj = d["loop"]
    assert hasattr(obj, "dataset")
    obj = obj.dataset
    return id(obj)


evaluation_batch_size_maximizer = MemoryUtilizationMaximizer(hasher=_hasher)


@evaluation_batch_size_maximizer
def _evaluate(
    loop: "EvaluationLoop",
    batch_size: int,
    use_tqdm: bool,
    tqdm_kwargs: OptionalKwargs,
    only_size_probing: bool = False,
    **kwargs,
) -> MetricResults:
    loop.model.eval()
    loader = loop.get_loader(batch_size=batch_size)
    total = len(loader)
    if only_size_probing:
        loader = itertools.islice(loader, 1)
        total = 1
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
    """An evaluation loop."""

    def __init__(
        self,
        model: Model,
        dataset: Dataset[BatchType],
        evaluator: Evaluator,
    ) -> None:
        """Initialize the evaluation loop."""
        self.model = model
        self.evaluator = evaluator
        self.dataset = dataset

    @abstractmethod
    def process_batch(self, batch: BatchType) -> None:
        """Process a single batch."""
        raise NotImplementedError

    def get_collator(self):
        """Get the collator to use for the data loader."""
        return None

    def get_loader(self, batch_size: int, **kwargs) -> DataLoader:
        """Create a data loader for a single evaluation round."""
        return DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
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
        """Evaluate."""
        if not batch_size:
            if self.model.device.type == "cpu":
                batch_size = 32
            else:
                batch_size = len(self.dataset)
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
        """Create index from dataframe."""
        key = [c for c in df.columns if c != target]
        triple_id_to_key_id = numpy.empty_like(df.index)
        indices = []
        bounds = [0]
        for key_id, (key, group) in enumerate(df.groupby(by=key)):
            unique_targets = group[target].unique()
            triple_id_to_key_id[group.index] = key_id
            indices.extend(unique_targets)
            bounds.append(len(indices))
        indices = cast(torch.LongTensor, torch.as_tensor(indices))
        bounds = numpy.asarray(bounds)
        return cls(triple_id_to_key_id=triple_id_to_key_id, bounds=bounds, indices=indices)

    def __getitem__(self, item: int) -> numpy.ndarray:
        key_id = self.triple_id_to_key_id[item]
        low, high = self.bounds[key_id : key_id + 2]
        return self.indices[low:high]


class LinkPredictionEvaluationDataset(Dataset):
    """A dataset for link prediction evaluation."""

    def __init__(
        self,
        factory: CoreTriplesFactory,
        targets: Optional[Collection[Target]] = None,
        filtered: bool = True,
    ) -> None:
        super().__init__()
        self.mapped_triples = factory.mapped_triples
        self.num_triples = factory.num_triples
        if targets is None:
            targets = [LABEL_HEAD, LABEL_TAIL]
        self.targets = list(targets)
        if filtered:
            df = pandas.DataFrame(data=factory.mapped_triples.numpy(), columns=[LABEL_HEAD, LABEL_RELATION, LABEL_TAIL])
            self.filter_indices = {target: FilterIndex.from_df(df=df, target=target) for target in targets}
        else:
            self.filter_indices = None

    @property
    def num_targets(self) -> int:
        """Return the number of targets."""
        return len(self.targets)

    def __len__(self) -> int:
        return self.num_triples * self.num_targets

    def __getitem__(self, index: int) -> Tuple[Target, MappedTriples, Optional[torch.LongTensor]]:
        target_id, index = divmod(index, self.num_triples)
        target = self.targets[target_id]
        nnz = None if self.filter_indices is None else self.filter_indices[target][index]
        return target, self.mapped_triples[index, :], nnz

    @staticmethod
    def collate(
        batch: Iterable[Tuple[Target, MappedTriples, Optional[torch.LongTensor]]]
    ) -> Mapping[Target, Tuple[MappedTriples, Optional[torch.Tensor]]]:
        """Collate batches."""
        triples: Mapping[Target, List[torch.LongTensor]] = defaultdict(list)
        nnz: Mapping[Target, List[torch.LongTensor]] = defaultdict(list)
        for target, triple, opt_nnz in batch:
            triples[target].append(triple)
            if opt_nnz is not None:
                nnz[target].append(opt_nnz)
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


class LinkPredictionEvaluationLoop(EvaluationLoop[Mapping[Target, MappedTriples]]):
    """Link prediction evaluation loop."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        evaluator: HintOrType[Evaluator] = None,
        evaluator_kwargs: OptionalKwargs = None,
        targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
        mode: Optional[InductiveMode] = None,
        **kwargs,
    ) -> None:
        """Initialize the evaluation loop."""
        evaluator = evaluator_resolver.make(evaluator, pos_kwargs=evaluator_kwargs)
        super().__init__(
            dataset=LinkPredictionEvaluationDataset(
                factory=triples_factory,
                targets=targets,
                filtered=evaluator.filtered or evaluator.requires_positive_mask,
            ),
            evaluator=evaluator,
            **kwargs,
        )
        self.targets = targets
        self.mode = mode

    def get_collator(self):  # noqa: D102
        return LinkPredictionEvaluationDataset.collate

    def process_batch(self, batch: Mapping[Target, MappedTriples]) -> None:  # noqa: D102
        for target, (hrt_batch, filter_batch) in batch.items():
            scores = self.model.predict(hrt_batch=hrt_batch, target=target, mode=self.mode)
            batch_ids = torch.arange(scores.shape[0], device=scores.device)
            target_ids = hrt_batch[:, TARGET_TO_INDEX[target]]
            true_scores = dense_positive_mask = None
            if self.evaluator.filtered:
                assert filter_batch is not None
                true_scores = scores[batch_ids, target_ids, None]
                # replace by nan
                scores = filter_scores_(scores=scores, filter_batch=filter_batch)
                # rewrite true scores
                scores[batch_ids, target_ids] = true_scores[:, 0]
            elif self.evaluator.requires_positive_mask:
                assert filter_batch is not None
                dense_positive_mask = torch.zeros_like(scores, dtype=torch.bool, device=filter_batch.device)
                dense_positive_mask[filter_batch[:, 0], filter_batch[:, 0]] = True
            self.evaluator.process_scores_(
                hrt_batch=hrt_batch,
                target=target,
                scores=scores,
                true_scores=true_scores,
                dense_positive_mask=dense_positive_mask,
            )
