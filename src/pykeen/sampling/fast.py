import abc
from typing import NamedTuple

import torch

from .negative_sampler import NegativeSampler


class FastSLCWABatch(NamedTuple):
    # shape: (batch_size, 3)
    hrt_pos: torch.LongTensor
    # shape: (batch_size, num_negs_h)
    h_neg: torch.LongTensor | None
    # shape: (batch_size, num_negs_r)
    r_neg: torch.LongTensor | None
    # shape: (batch_size, num_negs_t)
    t_neg: torch.LongTensor | None


class RestrictedNegativeSampler(NegativeSampler, abc.ABC):
    @abc.abstractmethod
    def corrupt_batch_structured(self, positive_batch: torch.LongTensor) -> FastSLCWABatch:
        raise NotImplementedError

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        raise NotImplementedError


class BasicRestrictedNegativeSampler(RestrictedNegativeSampler):
    def corrupt_batch_structured(self, positive_batch: torch.LongTensor) -> FastSLCWABatch:
        batch_size = positive_batch.shape[0]
        return FastSLCWABatch(
            hrt_pos=positive_batch,
            h_neg=torch.randint(
                high=self.num_entities,
                size=(batch_size, self.num_negs_per_pos),
                dtype=positive_batch.dtype,
                device=positive_batch.device,
            ),
            r_neg=torch.randint(
                high=self.num_relations,
                size=(batch_size, self.num_negs_per_pos),
                dtype=positive_batch.dtype,
                device=positive_batch.device,
            ),
            t_neg=torch.randint(
                high=self.num_entities,
                size=(batch_size, self.num_negs_per_pos),
                dtype=positive_batch.dtype,
                device=positive_batch.device,
            ),
        )
