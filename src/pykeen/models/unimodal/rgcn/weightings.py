# -*- coding: utf-8 -*-

"""Various edge weighting implementations for R-GCN."""

from typing import Callable, Collection, Mapping, Union

import torch

from ....utils import get_fn, normalize_string

__all__ = [
    'EdgeWeighting',
    'inverse_indegree_edge_weights',
    'inverse_outdegree_edge_weights',
    'symmetric_edge_weights',
    'edge_weightings',
    'get_edge_weighting',
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
_EDGE_WEIGHTINGS: Collection[EdgeWeighting] = [
    inverse_indegree_edge_weights,
    inverse_outdegree_edge_weights,
    symmetric_edge_weights,
]
_EDGE_WEIGHTINGS_SUFFIX = 'edge_weights'

edge_weightings: Mapping[str, EdgeWeighting] = {
    normalize_string(edge_weighting.__name__, suffix=_EDGE_WEIGHTINGS_SUFFIX): edge_weighting
    for edge_weighting in _EDGE_WEIGHTINGS
}


def get_edge_weighting(query: Union[None, str, EdgeWeighting]) -> EdgeWeighting:
    """Get the edge weighting."""
    return get_fn(
        query=query,
        default=inverse_indegree_edge_weights,
        lookup_dict=edge_weightings,
        suffix=_EDGE_WEIGHTINGS_SUFFIX,
    )
