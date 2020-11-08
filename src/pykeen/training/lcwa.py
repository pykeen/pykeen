# -*- coding: utf-8 -*-

"""Training KGE models based on the LCWA."""

import logging
from math import ceil
from typing import Optional, Tuple

import torch

from .training_loop import TrainingLoop
from .utils import apply_label_smoothing
from ..triples import LCWAInstances
from ..typing import MappedTriples

__all__ = [
    'LCWATrainingLoop',
]

logger = logging.getLogger(__name__)


class LCWATrainingLoop(TrainingLoop):
    """A training loop that uses the local closed world assumption training approach."""

    def _create_instances(self, use_tqdm: Optional[bool] = None) -> LCWAInstances:  # noqa: D102
        return self.triples_factory.create_lcwa_instances(use_tqdm=use_tqdm)

    @staticmethod
    def _get_batch_size(batch: Tuple[MappedTriples, torch.FloatTensor]) -> int:  # noqa: D102
        return batch[0].shape[0]

    def _process_batch(
        self,
        batch: Tuple[MappedTriples, torch.FloatTensor],
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
            predictions = self.model.score_t(hr_batch=batch_pairs, slice_size=slice_size)

        loss = self._loss_helper(
            predictions,
            batch_labels_full,
            label_smoothing,
        )
        return loss

    def _label_loss_helper(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: float,
    ) -> torch.FloatTensor:
        # Apply label smoothing
        if label_smoothing > 0.:
            labels = apply_label_smoothing(
                labels=labels,
                epsilon=label_smoothing,
                num_classes=self.model.num_entities,
            )

        return self.model._compute_loss(
            tensor_1=predictions,
            tensor_2=labels,
        )

    def _mr_loss_helper(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        # This shows how often one row has to be repeated
        repeat_rows = (labels == 1).nonzero(as_tuple=False)[:, 0]
        # Create boolean indices for negative labels in the repeated rows
        labels_negative = labels[repeat_rows] == 0
        # Repeat the predictions and filter for negative labels
        negative_scores = predictions[repeat_rows][labels_negative]

        # This tells us how often each true label should be repeated
        repeat_true_labels = (labels[repeat_rows] == 0).nonzero(as_tuple=False)[:, 0]
        # First filter the predictions for true labels and then repeat them based on the repeat vector
        positive_scores = predictions[labels == 1][repeat_true_labels]

        return self.model.compute_mr_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )

    def _self_adversarial_negative_sampling_loss_helper(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        _label_smoothing=None,
    ) -> torch.FloatTensor:
        """Compute self adversarial negative sampling loss."""
        # Split positive and negative scores
        positive_scores = predictions[labels == 1]
        negative_scores = predictions[labels == 0]

        return self.model.compute_self_adversarial_negative_sampling_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )

    def _slice_size_search(
        self,
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
