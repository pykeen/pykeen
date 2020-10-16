# -*- coding: utf-8 -*-

"""Various edge weighting implementations for R-GCN."""

from typing import Callable, Collection

import torch

__all__ = [
    'EdgeWeighting',
    'inverse_indegree_edge_weights',
    'inverse_outdegree_edge_weights',
    'symmetric_edge_weights',
    'edge_weightings',
]

EdgeWeighting = Callable[
    [torch.LongTensor, torch.LongTensor],
    torch.FloatTensor,
]


def inverse_indegree_edge_weights(source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
    """Normalize messages by inverse in-degree.

    :param source: shape: (num_edges,)
            The source indices.
    :param target: shape: (num_edges,)
        The target indices.

    :return: shape: (num_edges,)
         The edge weights.
    """
    # Calculate in-degree, i.e. number of incoming edges
    uniq, inv, cnt = torch.unique(target, return_counts=True, return_inverse=True)
    return cnt[inv].float().reciprocal()


def inverse_outdegree_edge_weights(source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
    """Normalize messages by inverse out-degree.

    :param source: shape: (num_edges,)
            The source indices.
    :param target: shape: (num_edges,)
        The target indices.

    :return: shape: (num_edges,)
         The edge weights.
    """
    # Calculate in-degree, i.e. number of incoming edges
    uniq, inv, cnt = torch.unique(source, return_counts=True, return_inverse=True)
    return cnt[inv].float().reciprocal()


def symmetric_edge_weights(source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
    """Normalize messages by product of inverse sqrt of in-degree and out-degree.

    :param source: shape: (num_edges,)
            The source indices.
    :param target: shape: (num_edges,)
        The target indices.

    :return: shape: (num_edges,)
         The edge weights.
    """
    return (
        inverse_indegree_edge_weights(source=source, target=target)
        * inverse_outdegree_edge_weights(source=source, target=target)
    ).sqrt()


#: A list of all implemented edge weightings for usage in HPO
edge_weightings: Collection[EdgeWeighting] = [
    inverse_indegree_edge_weights,
    inverse_outdegree_edge_weights,
    symmetric_edge_weights,
]
