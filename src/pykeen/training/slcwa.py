"""Training KGE models based on the sLCWA."""

import logging
from typing import Literal

from class_resolver import HintOrType, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from torch.utils.data import DataLoader

from .training_loop import TrainingLoop
from ..losses import Loss
from ..models.base import Model
from ..sampling import NegativeSampler
from ..triples import CoreTriplesFactory
from ..triples.instances import BaseBatchedSLCWAInstances, BatchedSLCWAInstances, SLCWABatch, SubGraphSLCWAInstances
from ..typing import FloatTensor, InductiveMode

__all__ = [
    "SLCWATrainingLoop",
]

logger = logging.getLogger(__name__)


class SLCWATrainingLoop(TrainingLoop[SLCWABatch]):
    """A training loop that uses the stochastic local closed world assumption training approach.

    [ruffinelli2020]_ call the sLCWA ``NegSamp`` in their work.
    """

    @update_docstring_with_resolver_keys(ResolverKey("negative_sampler", "pykeen.sampling.negative_sampler_resolver"))
    def __init__(
        self,
        *,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """Initialize the training loop.

        :param negative_sampler: The class, instance, or name of the negative sampler
        :param negative_sampler_kwargs: Keyword arguments to pass to the negative sampler class on instantiation
            for every positive one
        :param kwargs:
            Additional keyword-based parameters passed to TrainingLoop.__init__
        """
        super().__init__(**kwargs)
        self.negative_sampler = negative_sampler
        self.negative_sampler_kwargs = negative_sampler_kwargs

    # docstr-coverage: inherited
    def _create_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        *,
        sampler: Literal["schlichtkrull"] | None,
        batch_size: int,
        drop_last: bool,
        **kwargs,
    ) -> DataLoader[SLCWABatch]:  # noqa: D102
        cls: type[BaseBatchedSLCWAInstances]
        match sampler:
            case None:
                cls = BatchedSLCWAInstances
            case "schlichtkrull":
                cls = SubGraphSLCWAInstances
            case _:
                raise ValueError(f"Invalid {sampler=}")
        return DataLoader(
            dataset=cls.from_triples_factory(
                triples_factory,
                batch_size=batch_size,
                shuffle=kwargs.pop("shuffle", True),
                drop_last=drop_last,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=self.negative_sampler_kwargs,
                loss_weighter=self.loss_weighter,
                loss_weighter_kwargs=self.loss_weighter_kwargs,
            ),
            # disable automatic batching
            batch_size=None,
            batch_sampler=None,
            **kwargs,
        )

    @staticmethod
    # docstr-coverage: inherited
    def _get_batch_size(batch: SLCWABatch) -> int:  # noqa: D102
        return batch["positives"].shape[0]

    @staticmethod
    def _process_batch_static(
        model: Model,
        loss: Loss,
        mode: InductiveMode | None,
        batch: SLCWABatch,
        start: int | None,
        stop: int | None,
        label_smoothing: float = 0.0,
        slice_size: int | None = None,
    ) -> FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch = batch["positives"]
        negative_batch = batch["negatives"]
        positive_filter = batch.get("masks")
        pos_weights = batch.get("pos_weights")
        neg_weights = batch.get("neg_weights")

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
            positive_filter = positive_filter.to(model.device)
        if pos_weights is not None:
            pos_weights = pos_weights[start:stop].to(device=model.device)
        if neg_weights is not None:
            neg_weights = neg_weights[start:stop].to(device=model.device)
        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)

        # Compute negative and positive scores
        positive_scores = model.score_hrt(positive_batch, mode=mode)
        negative_scores = model.score_hrt(negative_batch, mode=mode).view(*negative_score_shape)

        return (
            loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=model._get_entity_len(mode=mode),
                pos_weights=pos_weights,
                neg_weights=neg_weights,
            )
            + model.collect_regularization_term()
        )

    # docstr-coverage: inherited
    def _process_batch(
        self,
        batch: SLCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: int | None = None,
    ) -> FloatTensor:  # noqa: D102
        return self._process_batch_static(
            model=self.model,
            loss=self.loss,
            mode=self.mode,
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
        )

    # docstr-coverage: inherited
    def _slice_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        batch_size: int,
        sub_batch_size: int,
        supports_sub_batching: bool,
    ):  # noqa: D102
        # Slicing is not possible for sLCWA
        if supports_sub_batching:
            report = "This model supports sub-batching, but it also requires slicing, which is not possible for sLCWA"
        else:
            report = "This model doesn't support sub-batching and slicing is not possible for sLCWA"
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
