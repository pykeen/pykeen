"""Training KGE models based on the LCWA."""

import logging
from collections.abc import Callable
from math import ceil
from typing import ClassVar

from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from torch_max_mem.api import is_oom_error

from .training_loop import TrainingLoop
from ..constants import get_target_column
from ..losses import Loss
from ..models import Model
from ..triples import CoreTriplesFactory, LCWAInstances
from ..triples.instances import LCWABatch
from ..typing import FloatTensor, InductiveMode, MappedTriples, TargetHint

__all__ = [
    "LCWATrainingLoop",
    "SymmetricLCWATrainingLoop",
]

logger = logging.getLogger(__name__)


class LCWATrainingLoop(TrainingLoop[LCWABatch]):
    r"""A training loop that is based upon the local closed world assumption (LCWA).

    Under the LCWA, for a given true training triple $(h, r, t) \in \mathcal{T}_{train}$, all triples
    $(h, r, t') \notin \mathcal{T}_{train}$ are assumed to be false. The training approach thus uses a 1-n scoring,
    where it efficiently computes scores for all triples $(h, r, t')$ for $t' \in \mathcal{E}$, i.e., sharing the
    same (head, relation)-pair.

    This implementation slightly generalizes the original LCWA, and allows to make the same assumption for relation, or
    head entity. In particular the second, i.e., predicting the relation, is commonly encountered in visual relation
    prediction.

    [ruffinelli2020]_ call the LCWA ``KvsAll`` in their work.
    """

    supports_slicing: ClassVar[bool] = True
    num_targets: int

    def __init__(self, *, target: TargetHint = None, **kwargs) -> None:
        """
        Initialize the training loop.

        :param target:
            The target column. Defaults to tail prediction.
        :param kwargs:
            Additional keyword-based parameters passed to TrainingLoop.__init__
        :raises ValueError:
            If an invalid target column is given
        """
        super().__init__(**kwargs)

        # normalize target column
        self.target = get_target_column(target)

        # The type inference is so confusing between the function switching
        # and polymorphism introduced by slicability that these need to be ignored
        if self.target == 0:
            self.score_method = self.model.score_h  # type: ignore
        elif self.target == 1:
            self.score_method = self.model.score_r  # type: ignore
        elif self.target == 2:
            self.score_method = self.model.score_t  # type: ignore
        else:
            raise ValueError(f"Invalid target column: {self.target}. Must be from {{0, 1, 2}}.")

        # Explicit mentioning of num_transductive_entities since in the evaluation there will be a different number
        # of total entities from another inductive inference factory
        self.num_targets = self.model.num_relations if self.target == 1 else self.model._get_entity_len(mode=self.mode)

    # docstr-coverage: inherited
    def _create_training_data_loader(
        self, triples_factory: CoreTriplesFactory, sampler: str | None, **kwargs
    ) -> DataLoader[LCWABatch]:  # noqa: D102
        if sampler:
            raise NotImplementedError(
                f"LCWA training does not support non-default batch sampling. Expected sampler=None, but got "
                f"sampler='{sampler}'.",
            )

        dataset = LCWAInstances.from_triples_factory(
            triples_factory,
            target=self.target,
            loss_weighter=self.loss_weighter,
            loss_weighter_kwargs=self.loss_weighter_kwargs,
        )
        return DataLoader(dataset=dataset, **kwargs)

    @staticmethod
    # docstr-coverage: inherited
    def _get_batch_size(batch: LCWABatch) -> int:  # noqa: D102
        return batch["pairs"].shape[0]

    @staticmethod
    def _process_batch_static(
        model: Model,
        score_method: Callable,
        loss: Loss,
        num_targets: int,
        mode: InductiveMode | None,
        batch: LCWABatch,
        start: int | None,
        stop: int | None,
        label_smoothing: float = 0.0,
        slice_size: int | None = None,
    ) -> FloatTensor:
        # Split batch components
        batch_pairs = batch["pairs"]
        batch_labels_full = batch["target"]
        batch_weights = batch.get("weights")

        # Send batch to device
        batch_pairs = batch_pairs[start:stop].to(device=model.device)
        batch_labels_full = batch_labels_full[start:stop].to(device=model.device)
        if batch_weights is not None:
            batch_weights = batch_weights[start:stop].to(device=model.device)

        predictions = score_method(batch_pairs, slice_size=slice_size, mode=mode)

        return (
            loss.process_lcwa_scores(
                predictions=predictions,
                labels=batch_labels_full,
                label_smoothing=label_smoothing,
                num_entities=num_targets,
                weights=batch_weights,
            )
            + model.collect_regularization_term()
        )

    # docstr-coverage: inherited
    def _process_batch(
        self,
        batch: LCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: int | None = None,
    ) -> FloatTensor:  # noqa: D102
        return self._process_batch_static(
            model=self.model,
            score_method=self.score_method,
            loss=self.loss,
            num_targets=self.num_targets,
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
    ) -> int:  # noqa: D102
        self._check_slicing_availability(supports_sub_batching)
        reached_max = False
        evaluated_once = False
        logger.info("Trying slicing now.")
        # Since the batch_size search with size 1, i.e. one tuple ((h, r) or (r, t)) scored on all entities,
        # must have failed to start slice_size search, we start with trying half the entities.
        slice_size = ceil(self.model.num_entities / 2)
        while True:
            try:
                logger.debug(f"Trying {slice_size=:_} now.")
                self._train(
                    triples_factory=triples_factory,
                    num_epochs=1,
                    batch_size=batch_size,
                    sub_batch_size=sub_batch_size,
                    slice_size=slice_size,
                    only_size_probing=True,
                )
            except RuntimeError as runtime_error:
                self._free_graph_and_cache()
                if not is_oom_error(runtime_error):
                    raise runtime_error
                if evaluated_once:
                    slice_size //= 2
                    logger.info(f"Concluded search with {slice_size=:_}.")
                    break
                if slice_size == 1:
                    raise MemoryError(
                        f"Even {slice_size=:_} doesn't fit into your memory with these parameters."
                    ) from runtime_error

                logger.debug(f"The {slice_size=:_} was too big, trying less now.")
                slice_size //= 2
                reached_max = True
            else:
                self._free_graph_and_cache()
                if reached_max:
                    logger.info(f"Concluded search with {slice_size=:_}.")
                    break
                slice_size *= 2
                evaluated_once = True

        return slice_size

    def _check_slicing_availability(self, supports_sub_batching: bool):
        if self.target == 0:
            return
        if self.target == 1:
            return
        if self.target == 2:
            return
        elif supports_sub_batching:
            report = (
                "This model supports sub-batching, but it also requires slicing,"
                " which is not implemented for this model yet."
            )
        else:
            report = "This model doesn't support sub-batching and slicing is not implemented for this model yet."
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")


# note: we use Tuple[Tensor] here, so we can re-use TensorDataset instead of having to create a custom one
class SymmetricLCWATrainingLoop(TrainingLoop[tuple[MappedTriples]]):
    r"""A "symmetric" LCWA scoring heads *and* tails at once.

    This objective was introduced by [lacroix2018]_ as

    .. math ::

        l_{i,j,k}(X) = - X_{i,j,k} + \log \left(
            \sum_{k'} \exp(X_{i,j,k′})
        \right) - X_{k,j+P,i} + \log \left(
            \sum_{i'} \exp (X_{k, j+P, i'})
        \right)


    which can be seen as a "symmetric LCWA", where for one batch of triples, we score both, heads *and* tails, given
    the remainder of the triple.

    .. note ::
        at the same time, there is a also a difference to the :class:`LCWATrainingLoop`: we do not group by e.g.,
        head+relation pairs. Thus, the name might be suboptimal and change in the future.
    """

    # docstr-coverage: inherited
    def _create_training_data_loader(
        self, triples_factory: CoreTriplesFactory, sampler: str | None, **kwargs
    ) -> DataLoader[tuple[MappedTriples]]:  # noqa: D102
        assert sampler is None
        return DataLoader(dataset=TensorDataset(triples_factory.mapped_triples), **kwargs)

    # docstr-coverage: inherited
    def _process_batch(
        self,
        batch: tuple[MappedTriples],
        start: int,
        stop: int,
        label_smoothing: float = 0,
        slice_size: int | None = None,
    ) -> FloatTensor:  # noqa: D102
        # unpack
        hrt_batch = batch[0]
        # Send batch to device
        hrt_batch = hrt_batch[start:stop].to(device=self.model.device)
        return (
            # head prediction
            self.loss.process_lcwa_scores(
                predictions=self.model.score_h(rt_batch=hrt_batch[:, 1:], slice_size=slice_size, mode=self.mode),
                # TODO: exploit sparsity
                # note: this is different to what we do for LCWA, where we collect *all* training entities
                #   for which the combination is true
                labels=functional.one_hot(hrt_batch[:, 0], num_classes=self.model.num_entities).float(),
                label_smoothing=label_smoothing,
                num_entities=self.model.num_entities,
            )
            # tail prediction
            + self.loss.process_lcwa_scores(
                predictions=self.model.score_t(hr_batch=hrt_batch[:, :-1], slice_size=slice_size, mode=self.mode),
                # TODO: exploit sparsity
                labels=functional.one_hot(hrt_batch[:, 2], num_classes=self.model.num_entities).float(),
                label_smoothing=label_smoothing,
                num_entities=self.model.num_entities,
            )
            # regularization
            + self.model.collect_regularization_term()
        )

    @staticmethod
    # docstr-coverage: inherited
    def _get_batch_size(batch: tuple[MappedTriples]) -> int:  # noqa: D102
        assert len(batch) == 1
        return batch[0].shape[0]

    def _slice_size_search(self, **kwargs) -> int:
        # TODO?
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
