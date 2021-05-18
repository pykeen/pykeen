# -*- coding: utf-8 -*-

"""Training KGE models based on the LCWA."""

import logging
from math import ceil
from typing import Optional

import torch

from .training_loop import TrainingLoop
from ..triples import CoreTriplesFactory, Instances
from ..triples.instances import LCWABatchType, LCWASampleType

__all__ = [
    'LCWATrainingLoop',
]

logger = logging.getLogger(__name__)


class LCWATrainingLoop(TrainingLoop[LCWASampleType, LCWABatchType]):
    """A training loop that uses the local closed world assumption training approach."""

    def _create_instances(self, triples_factory: CoreTriplesFactory) -> Instances:  # noqa: D102
        return triples_factory.create_lcwa_instances()

    @staticmethod
    def _get_batch_size(batch: LCWABatchType) -> int:  # noqa: D102
        return batch[0].shape[0]

    def _process_batch(
        self,
        batch: LCWABatchType,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Split batch components
        batch_pairs, batch_labels_full = batch

        # Send batch to device
        batch_pairs = batch_pairs[start:stop].to(device=self.device)
        batch_labels_full = batch_labels_full[start:stop].to(device=self.device)

        if slice_size is None:
            predictions = self.model.score_t(hr_batch=batch_pairs)
        else:
            predictions = self.model.score_t(hr_batch=batch_pairs, slice_size=slice_size)  # type: ignore

        return self.loss.process_lcwa_scores(
            predictions=predictions,
            labels=batch_labels_full,
            label_smoothing=label_smoothing,
            num_entities=self.model.num_entities,
        ) + self.model.collect_regularization_term()

    def _slice_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        training_instances: Instances,
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
                logger.debug(f'Trying slice size {slice_size} now.')
                self._train(
                    triples_factory=triples_factory,
                    training_instances=training_instances,
                    num_epochs=1,
                    batch_size=batch_size,
                    sub_batch_size=sub_batch_size,
                    slice_size=slice_size,
                    only_size_probing=True,
                )
            except RuntimeError as e:
                self._free_graph_and_cache()
                if 'CUDA out of memory.' not in e.args[0]:
                    raise e
                if evaluated_once:
                    slice_size //= 2
                    logger.info(f'Concluded search with slice_size {slice_size}.')
                    break
                if slice_size == 1:
                    raise MemoryError(
                        f"Even slice_size={slice_size} doesn't fit into your memory with these"
                        f" parameters.",
                    ) from e

                logger.debug(
                    f'The slice_size {slice_size} was too big, trying less now.',
                )
                slice_size //= 2
                reached_max = True
            else:
                self._free_graph_and_cache()
                if reached_max:
                    logger.info(f'Concluded search with slice_size {slice_size}.')
                    break
                slice_size *= 2
                evaluated_once = True

        return slice_size

    def _check_slicing_availability(self, supports_sub_batching: bool):
        if self.model.can_slice_t:
            return
        elif supports_sub_batching:
            report = (
                "This model supports sub-batching, but it also requires slicing,"
                " which is not implemented for this model yet."
            )
        else:
            report = (
                "This model doesn't support sub-batching and slicing is not"
                " implemented for this model yet."
            )
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
