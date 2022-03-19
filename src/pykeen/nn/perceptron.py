# -*- coding: utf-8 -*-

"""Perceptron-like modules."""

from typing import Union

import torch
from torch import nn

__all__ = [
    "ConcatMLP",
]


class ConcatMLP(nn.Sequential):
    """A 2-layer MLP with ReLU activation and dropout applied to the concatenation of token representations.

    This is for conveniently choosing a configuration similar to the paper. For more complex aggregation mechanisms,
    pass an arbitrary callable instead.

    .. seealso:: https://github.com/migalkin/NodePiece/blob/d731c9990/lp_rp/pykeen105/nodepiece_rotate.py#L57-L65
    """

    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        dropout: float = 0.1,
        ratio: Union[int, float] = 2,
    ):
        """Initialize the module.

        :param num_tokens:
            the number of tokens
        :param embedding_dim:
            the embedding dimension for a single token
        :param dropout:
            the dropout value on the hidden layer
        :param ratio:
            the ratio of the embedding dimension to the hidden layer size.
        """
        hidden_dim = int(ratio * embedding_dim)
        super().__init__(
            nn.Linear(num_tokens * embedding_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, xs: torch.FloatTensor, dim: int) -> torch.FloatTensor:
        """Forward the MLP on the given dimension.

        :param xs: The tensor to forward
        :param dim: Only a parameter to match the signature of torch.mean / torch.sum
            this class is not thought to be usable from outside
        :returns: The tensor after applying this MLP
        """
        assert dim == -2
        return super().forward(xs.view(*xs.shape[:-2], -1))
