"""Utilities for NodePiece."""

import logging
from typing import Collection, Mapping

import torch
from tqdm.auto import tqdm

__all__ = [
    "random_sample_no_replacement",
]

logger = logging.getLogger(__name__)


def random_sample_no_replacement(
    pool: Mapping[int, Collection[int]],
    num_tokens: int,
) -> torch.LongTensor:
    """Sample randomly without replacement num_tokens relations for each entity."""
    assignment = torch.full(
        size=(len(pool), num_tokens),
        dtype=torch.long,
        fill_value=-1,
    )
    # TODO: vectorization?
    for idx, this_pool in tqdm(pool.items(), desc="sampling", leave=False, unit_scale=True):
        this_pool_t = torch.as_tensor(data=list(this_pool), dtype=torch.long)
        this_pool = this_pool_t[torch.randperm(this_pool_t.shape[0])[:num_tokens]]
        assignment[idx, : len(this_pool_t)] = this_pool
    return assignment
