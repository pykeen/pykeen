# -*- coding: utf-8 -*-

"""Evaluation loops for KGE models."""

import itertools
from abc import abstractmethod
from typing import Collection, Generic, Optional, TypeVar

import torch
from class_resolver import OptionalKwargs
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from .evaluator import Evaluator, MetricResults
from ..constants import TARGET_TO_INDEX
from ..models import Model
from ..triples import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_TAIL, MappedTriples, Target

BatchType = TypeVar("BatchType")


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
        raise NotImplementedError

    def get_loader(self, batch_size: int, **kwargs) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False, pin_memory=True, **kwargs)

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
        return self._evaluate(batch_size=batch_size, use_tqdm=use_tqdm, tqdm_kwargs=tqdm_kwargs, **kwargs)

    def _evaluate(
        self,
        batch_size: int,
        use_tqdm: bool,
        tqdm_kwargs: OptionalKwargs,
        only_size_probing: bool,
        **kwargs,
    ) -> MetricResults:
        self.model.eval()
        loader = self.get_loader(batch_size=batch_size)
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
                **tqdm_kwargs,
            )
        for batch in loader:
            self.process_batch(batch=batch)
        return self.evaluator.finalize()


class LinkPredictionEvaluationDataset(Dataset):
    def __init__(self, factory: CoreTriplesFactory) -> None:
        super().__init__()
        self.mapped_triples = factory.mapped_triples
        self.num_triples = factory.num_triples

    def __len__(self) -> int:
        return self.num_triples

    def __getitem__(self, index: int) -> MappedTriples:
        return self.mapped_triples[index, :]


class LinkPredictionEvaluationLoop(EvaluationLoop[MappedTriples]):
    """Link prediction evaluation loop."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
        **kwargs,
    ) -> None:
        super().__init__(dataset=LinkPredictionEvaluationDataset(factory=triples_factory), **kwargs)
        self.targets = targets

    def process_batch(self, batch: MappedTriples) -> None:  # noqa: D102
        hrt_batch = batch
        for target in self.targets:
            scores = self.model.predict(hrt_batch=hrt_batch, target=target)
            true_scores = scores[torch.arange(scores.shape[0]), hrt_batch[:, TARGET_TO_INDEX[target]], None]
            self.evaluator.process_scores_(
                hrt_batch=hrt_batch,
                target=target,
                scores=scores,
                true_scores=true_scores,
                dense_positive_mask=None,
            )
