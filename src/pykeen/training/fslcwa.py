# -*- coding: utf-8 -*-

"""Training KGE models based on the sLCWA."""
import logging
from typing import Optional

import torch.utils.data
from torch.utils.data import DataLoader

from .training_loop import TrainingLoop
from ..models.base import split_batch_for_prediction
from ..sampling.fast import FastSLCWABatch
from ..triples import CoreTriplesFactory
from ..triples.instances import SLCWABatch
from ..typing import LABEL_RELATION, LABEL_HEAD, LABEL_TAIL

__all__ = [
    "FastSLCWATrainingLoop",
]


logger = logging.getLogger(__name__)


class FastSLCWATrainingLoop(TrainingLoop[FastSLCWABatch, FastSLCWABatch]):

    def __init__(self, num_negs_per_pos: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_negs_per_pos = num_negs_per_pos

    def _create_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        batch_size: int,
        drop_last: bool,
        num_workers: int,
        pin_memory: bool,
        sampler: Optional[str],
    ) -> DataLoader[SLCWABatch]:  # noqa: D102
        return DataLoader(
            dataset=triples_factory.create_fslcwa_instances(
                batch_size=batch_size,
                drop_last=drop_last,
                num_negs_per_pos=self.num_negs_per_pos,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            # disable automatic batching
            batch_size=None,
            batch_sampler=None,
        )

    @staticmethod
    def _get_batch_size(batch: FastSLCWABatch) -> int:  # noqa: D102
        return batch.hrt_pos.shape[0]

    def _process_batch(
        self,
        batch: FastSLCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        loss = 0.0
        for target, negatives in {
            LABEL_HEAD: batch.h_neg,
            LABEL_RELATION: batch.r_neg,
            LABEL_TAIL: batch.t_neg,
        }.items():
            if negatives is None:
                continue
            base, positive = split_batch_for_prediction(batch.hrt_pos, target=target)
            scores = self.model.score(
                batch=base,
                target=target,
                restriction=torch.cat([positive[:, None], negatives], dim=1),
                slice_size=slice_size,
                mode=self.mode,
            )
            positive_scores = scores[:, :1]
            negative_scores = scores[:, 1:]
            loss = loss + self.loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=None,
                num_entities=(
                    self.model.num_relations if target == LABEL_RELATION else self.model._get_entity_len(mode=self.mode)
                ),
            )
        return loss + self.model.collect_regularization_term()

    def _slice_size_search(
        self, *, triples_factory: CoreTriplesFactory, batch_size: int, sub_batch_size: int, supports_sub_batching: bool
    ) -> int:
        raise NotImplementedError
