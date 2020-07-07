# -*- coding: utf-8 -*-

"""Embedding weight initialization routines."""

import math

from torch import nn
from torch.nn import init

__all__ = [
    'embedding_xavier_uniform_',
    'embedding_xavier_normal_',
]


def embedding_xavier_uniform_(embedding: nn.Embedding, gain: float = 1.) -> nn.Embedding:
    r"""Initialize weights of embedding.

    Proceed as if it was a linear layer with fan_in of zero and Xavier uniform
    initialization is used, i.e. fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{embedding_dim}}}

    Similar to Glorot/Xavier initialization.

    :param embedding: An embedding
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier uniform initializer.

    Examples:

    >>> e = nn.Embedding(num_embeddings=3, embedding_dim=5)
    >>> embedding_xavier_uniform_(embedding=e, gain=nn.init.calculate_gain('relu'))

    """
    bound = gain * 6 / math.sqrt(embedding.embedding_dim)
    init.uniform_(embedding.weight, -bound, bound)
    return embedding


def embedding_xavier_normal_(embedding: nn.Embedding, gain: float = 1.) -> nn.Embedding:
    r"""Initialize weights of embedding.

    Proceed as if it was a linear layer with fan_in of zero and Xavier normal
    initialization is used. Fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{N}(0, a^2)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{2}{\text{embedding_dim}}}

    Similar to Glorot/Xavier initialization.

    :param embedding: An embedding
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier normal initializer.

    Examples:

    >>> e = nn.Embedding(num_embeddings=3, embedding_dim=5)
    >>> embedding_xavier_normal_(embedding=e, gain=nn.init.calculate_gain('relu'))

    """
    std = gain * 2 / math.sqrt(embedding.embedding_dim)
    init.normal_(embedding.weight, mean=0., std=std)
    return embedding
