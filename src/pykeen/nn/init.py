# -*- coding: utf-8 -*-

"""Embedding weight initialization routines."""

import math
from typing import Union

import torch.nn
import torch.nn.init
from torch.nn import functional

from .emb import Embedding

__all__ = [
    'xavier_uniform_',
    'xavier_uniform_normed_',
    'xavier_normal_',
    'embedding_xavier_uniform_',
    'embedding_xavier_normal_',
]


def xavier_uniform_(tensor, gain: float = 1.):
    """Initialize weights of the tensor similarly to Glorot/Xavier initialization."""
    bound = gain * 6 / math.sqrt(tensor.shape[1])  # TODO @mberr is that index right?
    torch.nn.init.uniform_(tensor, -bound, bound)
    return tensor


def xavier_uniform_normed_(tensor: torch.Tensor, gain: float = 1.) -> torch.Tensor:
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization the normalize.

    Proceed as if it was a linear layer with fan_in of zero and Xavier uniform
    initialization is used, i.e. fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{U}(-a, a)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{6}{\text{embedding_dim}}}

    :param tensor: A tensor
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier uniform initializer.
    """
    x = xavier_uniform_(tensor, gain=gain)
    functional.normalize(x.data, out=x.data)
    return x


# TODO delete this function and move documentation
def embedding_xavier_uniform_(embedding: Union[torch.nn.Embedding, Embedding], gain: float = 1.):
    """Initialize weights of embedding similarly to Glorot/Xavier initialization.

    :param embedding: An embedding
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier uniform initializer.

    In the following example, an embedding is initialized using the suggested gain for the rectified
    linear unit (ReLu).

    >>> import pykeen.nn
    >>> from pykeen.nn.init import embedding_xavier_uniform_
    >>> from torch.nn.init import calculate_gain
    >>> e = pykeen.nn.Embedding(num_embeddings=3, embedding_dim=5)
    >>> embedding_xavier_uniform_(embedding=e, gain=calculate_gain('relu'))

    """
    if isinstance(embedding, Embedding):
        return xavier_uniform_(embedding._embeddings.weight, gain=gain)
    elif isinstance(embedding, torch.nn.Embedding):
        return xavier_uniform_(embedding.weight, gain=gain)
    else:
        raise TypeError


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Initialize weights of the tensor similarly to Glorot/Xavier initialization.

    Proceed as if it was a linear layer with fan_in of zero and Xavier normal
    initialization is used. Fill the weight of input `embedding` with values values
    sampled from :math:`\mathcal{N}(0, a^2)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{2}{\text{embedding_dim}}}

    :param tensor: A tensor
    :param gain: An optional scaling factor, defaults to 1.0.
    :return: Embedding with weights by the Xavier normal initializer.
    """
    std = gain * 2 / math.sqrt(tensor.shape[1])  # TODO @mberr is that index right?
    torch.nn.init.normal_(tensor, mean=0., std=std)
    return tensor


# TODO delete this function and move documentation
def embedding_xavier_normal_(embedding: Union[torch.nn.Embedding, Embedding], gain: float = 1.):
    """Initialize weights of embedding similarly to Glorot/Xavier initialization.

    In the following example, an embedding is initialized using the suggested gain for the rectified
    linear unit (ReLu).

    >>> import pykeen.nn
    >>> from pykeen.nn.init import embedding_xavier_normal_
    >>> from torch.nn.init import calculate_gain
    >>> e = pykeen.nn.Embedding(num_embeddings=3, embedding_dim=5)
    >>> embedding_xavier_normal_(embedding=e, gain=calculate_gain('relu'))

    """
    if isinstance(embedding, Embedding):
        return xavier_normal_(embedding._embeddings.weight, gain=gain)
    elif isinstance(embedding, torch.nn.Embedding):
        return xavier_normal_(embedding.weight, gain=gain)
    else:
        raise TypeError
