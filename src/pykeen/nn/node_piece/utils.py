"""Utilities for NodePiece."""

import logging
from typing import Collection, Mapping, Optional

import numpy
import torch
from tqdm.auto import tqdm

__all__ = [
    "random_sample_no_replacement",
    "ensure_num_entities",
    "prepare_edges_for_metis",
]

logger = logging.getLogger(__name__)


def random_sample_no_replacement(
    pool: Mapping[int, Collection[int]],
    num_tokens: int,
    num_entities: Optional[int] = None,
) -> torch.LongTensor:
    """Sample randomly without replacement num_tokens relations for each entity.

    If a graph has disconnected nodes, then num_entities > number of rows in the pool.

    :param pool:
        a dictionary of entity: [relations]
    :param num_tokens:
        the number of tokens to sample for each entity
    :param num_entities:
        the total number of nodes in the graph, might be bigger than the pool size for graphs with disconnected nodes.
        If not given, is calculated based the length of ``pool``.

    :return: shape: (num_entities, num_tokens), -1 <= res < vocabulary_size
        the selected relation IDs for each entity. -1 is used as a padding token.
    """
    if num_entities is None:
        num_entities = len(pool)
    assignment = torch.full(
        size=(num_entities, num_tokens),
        dtype=torch.long,
        fill_value=-1,
    )
    # TODO: vectorization?
    for idx, this_pool in tqdm(pool.items(), desc="sampling", leave=False, unit_scale=True):
        this_pool_t = torch.as_tensor(data=list(this_pool), dtype=torch.long)
        this_pool = this_pool_t[torch.randperm(this_pool_t.shape[0])[:num_tokens]]
        assignment[idx, : len(this_pool_t)] = this_pool
    return assignment


def ensure_num_entities(edge_index: numpy.ndarray, num_entities: Optional[int] = None) -> int:
    """Calculate the number of entities from the edge index if not given."""
    if num_entities is not None:
        return num_entities
    return edge_index.max().item() + 1


def prepare_edges_for_metis(edge_index: torch.Tensor) -> torch.LongTensor:
    """Prepare the edge index for METIS partitioning to prevent segfaults."""
    # remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    # add inverse edges and remove duplicates
    return torch.cat([edge_index, edge_index.flip(0)], dim=-1).unique(dim=1)
