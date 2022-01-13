# -*- coding: utf-8 -*-

"""Evaluation loops for KGE models."""

from abc import abstractmethod
import math
from typing import Generic, Optional, Tuple, TypeVar
import itertools

from class_resolver import OptionalKwargs
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from pykeen.typing import MappedTriples


from ..models import Model
from ..triples import CoreTriplesFactory

BatchType = TypeVar("BatchType")
ResultType = TypeVar("ResultType")


def _get_next_power_of_two(x: int) -> int:
    return 2 ** int(math.ceil(math.log2(x)))


class EvaluationResultAggregator(Generic[BatchType, ResultType]):
    @abstractmethod
    def process_batch(self, model: Model, batch: BatchType) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> ResultType:
        raise NotImplementedError


class EvaluationLoop(Generic[BatchType, ResultType]):
    """An evaluation loop."""

    def __init__(
        self,
        model: Model,
        automatic_memory_optimization: bool = True,
    ) -> None:
        """Initialize the evaluation loop.

        :param model: The model to evaluate
        :param triples_factory: The evaluation triples factory
        :param automatic_memory_optimization: bool
            Whether to automatically optimize the (sub-)batch size during evaluation with regards
            to the hardware at hand.
        """
        self.model = model
        self.dataset = self._create_dataset()
        self.automatic_memory_optimization = automatic_memory_optimization

    @abstractmethod
    def _create_dataset(self) -> Dataset[BatchType]:
        raise NotImplementedError

    @abstractmethod
    def _create_aggregator(self) -> EvaluationResultAggregator[BatchType, ResultType]:
        raise NotImplementedError

    def _create_data_loader(
        self,
        dataset: Dataset[BatchType],
        batch_size: int,
        **kwargs,
    ) -> DataLoader[BatchType]:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
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
    ):
        if batch_size is None:
            if not self.automatic_memory_optimization:
                raise ValueError
            if self.model.device.type == "cpu":
                batch_size = 32
            else:
                batch_size = _get_next_power_of_two(len(self.dataset))
        # TODO: AMO
        return self._evaluate(batch_size=batch_size, use_tqdm=use_tqdm, tqdm_kwargs=tqdm_kwargs, **kwargs)

    def _evaluate(
        self,
        batch_size: int,
        use_tqdm: bool,
        tqdm_kwargs: OptionalKwargs,
        only_size_probing: bool,
        **kwargs,
    ):
        self.model.eval()
        loader = self._create_data_loader(
            dataset=self.dataset,
            batch_size=batch_size,
            **kwargs,
        )
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
        aggregator = self._create_aggregator()
        for batch in loader:
            aggregator.process_batch(model=self.model, batch=batch)
        return aggregator.finalize()


class LinkPredictionEvaluationDataset(Dataset):
    def __init__(self, factory: CoreTriplesFactory) -> None:
        super().__init__()
        self.mapped_triples = factory.mapped_triples
        self.num_triples = factory.num_triples

    def __len__(self) -> int:
        return self.num_triples

    def __getitem__(self, index: int) -> torch.LongTensor:
        # TODO: filtering
        return self.mapped_triples[index, :]


class LinkPredictionAggregator(EvaluationResultAggregator[MappedTriples, ResultType]):
    def __init__(self) -> None:
        super().__init__()
        self.ranks = ...  # cf. RankBasedEvaluator

    def process_batch(self, model: Model, batch: MappedTriples) -> None:
        hr_batch = batch[:, :2]
        scores = model.predict_t(hr_batch=hr_batch)
        ranks = ...

        rt_batch = batch[:, 1:]
        scores = model.predict_h(rt_batch=rt_batch)
        ranks = ...

    def finalize(self) -> ResultType:
        ...


class LinkPredictionEvaluationLoop(EvaluationLoop):
    """Link prediction evaluation loop."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        **kwargs,
    ) -> None:
        self.triples_factory = triples_factory
        super().__init__(**kwargs)

    def _create_dataset(self) -> Dataset[MappedTriples]:
        return LinkPredictionEvaluationDataset(factory=self.triples_factory)

    def _create_aggregator(self) -> LinkPredictionAggregator:
        return LinkPredictionAggregator()
