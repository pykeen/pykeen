# -*- coding: utf-8 -*-

"""Training inductive KGE models based on the LCWA."""

import logging
from typing import Optional

import torch

from .lcwa import LCWATrainingLoop
from ..triples.instances import LCWABatchType

__all__ = [
    "InductiveLCWATrainingLoop",
]

logger = logging.getLogger(__name__)

name_to_index = {name: index for index, name in enumerate("hrt")}


class InductiveLCWATrainingLoop(LCWATrainingLoop):
    """
    Inductive LCWA loop. The main difference: explicit setting of the mode=train argument.
    This is necessary since in the evaluation (valid / test) training factory will be different, and depending on the
    "mode" param, a model should invoke a proper inference graph
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Explicit mentioning of num_transductive_entities since in the evaluation there will be a different number
        # of total entities from another inductive inference factory
        self.num_targets = self.model.num_relations if self.target == 1 else self.model.num_train_entities

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
            predictions = self.score_method(batch_pairs, mode="train")
        else:
            predictions = self.score_method(batch_pairs, slice_size=slice_size, mode="train")  # type: ignore

        return (
            self.loss.process_lcwa_scores(
                predictions=predictions,
                labels=batch_labels_full,
                label_smoothing=label_smoothing,
                num_entities=self.num_targets,
            )
            + self.model.collect_regularization_term()
        )
