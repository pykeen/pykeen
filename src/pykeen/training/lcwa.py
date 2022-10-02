# -*- coding: utf-8 -*-

"""Training KGE models based on the LCWA."""

import logging
from math import ceil
from typing import Callable, Optional, Tuple, Union

import torch
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from .training_loop import TrainingLoop
from ..losses import Loss
from ..models import Model
from ..triples import CoreTriplesFactory
from ..triples.instances import LCWABatchType, LCWASampleType
from ..typing import InductiveMode, MappedTriples

__all__ = [
    "LCWATrainingLoop",
    "SymmetricLCWATrainingLoop",
]

logger = logging.getLogger(__name__)

name_to_index = {name: index for index, name in enumerate("hrt")}


class LCWATrainingLoop(TrainingLoop[LCWASampleType, LCWABatchType]):
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

    def __init__(
        self,
        *,
        target: Union[None, str, int] = None,
        **kwargs,
    ):
        """
        Initialize the training loop.

        :param target:
            The target column. From {0, 1, 2} for head/relation/tail prediction. Defaults to 2, i.e., tail prediction.
        :param kwargs:
            Additional keyword-based parameters passed to TrainingLoop.__init__
        :raises ValueError:
            If an invalid target column is given
        """
        super().__init__(**kwargs)

        # normalize target column
        if target is None:
            target = 2
        if isinstance(target, str):
            target = name_to_index[target]
        self.target = target

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
        self, triples_factory: CoreTriplesFactory, sampler: Optional[str], **kwargs
    ) -> DataLoader[LCWABatchType]:  # noqa: D102
        if sampler:
            raise NotImplementedError(
                f"LCWA training does not support non-default batch sampling. Expected sampler=None, but got "
                f"sampler='{sampler}'.",
            )

        dataset = triples_factory.create_lcwa_instances(target=self.target)
        return DataLoader(dataset=dataset, collate_fn=dataset.get_collator(), **kwargs)

    @staticmethod
    # docstr-coverage: inherited
    def _get_batch_size(batch: LCWABatchType) -> int:  # noqa: D102
        return batch[0].shape[0]

    @staticmethod
    def _process_batch_static(
        model: Model,
        score_method: Callable,
        loss: Loss,
        num_targets: Optional[int],
        mode: Optional[InductiveMode],
        batch: LCWABatchType,
        start: Optional[int],
        stop: Optional[int],
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Split batch components
        batch_pairs, batch_labels_full = batch

        # Send batch to device
        batch_pairs = batch_pairs[start:stop].to(device=model.device)
        batch_labels_full = batch_labels_full[start:stop].to(device=model.device)

        predictions = score_method(batch_pairs, slice_size=slice_size, mode=mode)

        return (
            loss.process_lcwa_scores(
                predictions=predictions,
                labels=batch_labels_full,
                label_smoothing=label_smoothing,
                num_entities=num_targets,
            )
            + model.collect_regularization_term()
        )

    # docstr-coverage: inherited
    def _process_batch(
        self,
        batch: LCWABatchType,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
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
                logger.debug(f"Trying slice size {slice_size} now.")
                self._train(
                    triples_factory=triples_factory,
                    num_epochs=1,
                    batch_size=batch_size,
                    sub_batch_size=sub_batch_size,
                    slice_size=slice_size,
                    only_size_probing=True,
                )
            except RuntimeError as e:
                self._free_graph_and_cache()
                if "CUDA out of memory." not in e.args[0]:
                    raise e
                if evaluated_once:
                    slice_size //= 2
                    logger.info(f"Concluded search with slice_size {slice_size}.")
                    break
                if slice_size == 1:
                    raise MemoryError(
                        f"Even slice_size={slice_size} doesn't fit into your memory with these" f" parameters.",
                    ) from e

                logger.debug(
                    f"The slice_size {slice_size} was too big, trying less now.",
                )
                slice_size //= 2
                reached_max = True
            else:
                self._free_graph_and_cache()
                if reached_max:
                    logger.info(f"Concluded search with slice_size {slice_size}.")
                    break
                slice_size *= 2
                evaluated_once = True

        return slice_size

    def _check_slicing_availability(self, supports_sub_batching: bool):
        if self.target == 0 and self.model.can_slice_h:
            return
        if self.target == 1 and self.model.can_slice_r:
            return
        if self.target == 2 and self.model.can_slice_t:
            return
        elif supports_sub_batching:
            report = (
                "This model supports sub-batching, but it also requires slicing,"
                " which is not implemented for this model yet."
            )
        else:
            report = "This model doesn't support sub-batching and slicing is not" " implemented for this model yet."
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")


# note: we use Tuple[Tensor] here, so we can re-use TensorDataset instead of having to create a custom one
class SymmetricLCWATrainingLoop(TrainingLoop[Tuple[MappedTriples], Tuple[MappedTriples]]):
    r"""
    A "symmetric" LCWA scoring heads *and* tails at once.

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
        self, triples_factory: CoreTriplesFactory, sampler: Optional[str], **kwargs
    ) -> DataLoader[Tuple[MappedTriples]]:  # noqa: D102
        assert sampler is None
        return DataLoader(dataset=TensorDataset(triples_factory.mapped_triples), **kwargs)

    # docstr-coverage: inherited
    def _process_batch(
        self,
        batch: Tuple[MappedTriples],
        start: int,
        stop: int,
        label_smoothing: float = 0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
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
    def _get_batch_size(batch: Tuple[MappedTriples]) -> int:  # noqa: D102
        assert len(batch) == 1
        return batch[0].shape[0]

    def _slice_size_search(self, **kwargs) -> int:
        # TODO?
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
