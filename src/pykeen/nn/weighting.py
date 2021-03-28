# -*- coding: utf-8 -*-

"""Various edge weighting implementations for R-GCN."""

from abc import abstractmethod

import torch
from class_resolver import Resolver
from torch import nn

__all__ = [
    "EdgeWeighting",
    'InverseInDegreeEdgeWeighting',
    'InverseOutDegreeEdgeWeighting',
    'SymmetricEdgeWeighting',
    "edge_weight_resolver",
]


class EdgeWeighting(nn.Module):
    """Base class for edge weightings."""

    @abstractmethod
    def forward(self, source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
        """Compute edge weights.

        :param source: shape: (num_edges,)
                The source indices.
        :param target: shape: (num_edges,)
            The target indices.

        :return: shape: (num_edges,)
             The edge weights.
        """
        raise NotImplementedError


def _inverse_frequency_weighting(idx: torch.LongTensor) -> torch.FloatTensor:
    """Calculate inverse relative frequency weighting."""
    # Calculate in-degree, i.e. number of incoming edges
    inv, cnt = torch.unique(idx, return_counts=True, return_inverse=True)[1:]
    return cnt[inv].float().reciprocal()


class InverseInDegreeEdgeWeighting(EdgeWeighting):
    """Normalize messages by inverse in-degree."""

    def forward(self, source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return _inverse_frequency_weighting(idx=target)


class InverseOutDegreeEdgeWeighting(EdgeWeighting):
    """Normalize messages by inverse out-degree."""

    def forward(self, source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return _inverse_frequency_weighting(idx=source)


class SymmetricEdgeWeighting(EdgeWeighting):
    """Normalize messages by product of inverse sqrt of in-degree and out-degree."""

    def forward(self, source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return (_inverse_frequency_weighting(idx=source) * _inverse_frequency_weighting(idx=target)).sqrt()


edge_weight_resolver = Resolver.from_subclasses(base=EdgeWeighting, default=SymmetricEdgeWeighting)
