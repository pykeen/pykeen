# -*- coding: utf-8 -*-

"""Embedding modules."""

from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import torch
import torch.nn
from torch import nn
from torch.nn import functional

from .init import init_phases, xavier_normal_, xavier_uniform_
from .norm import complex_normalize
from ..regularizers import Regularizer
from ..typing import Constrainer, Hint, Initializer, Normalizer
from ..utils import clamp_norm, compose

__all__ = [
    'RepresentationModule',
    'Embedding',
    'EmbeddingSpecification',
]


class RepresentationModule(nn.Module, ABC):
    """
    A base class for obtaining representations for entities/relations.

    A representation module maps integer IDs to representations, which are tensors of floats.

    `max_id` defines the upper bound of indices we are allowed to request (exclusively). For simple embeddings this is
    equivalent to num_embeddings, but more a more appropriate word for general non-embedding representations, where the
    representations could come from somewhere else, e.g. a GNN encoder.

    `shape` describes the shape of a single representation. In case of a vector embedding, this is just a single
    dimension. For others, e.g. :class:`pykeen.models.RESCAL`, we have 2-d representations, and in general it can be
    any fixed shape.

    We can look at all representations as a tensor of shape `(max_id, *shape)`, and this is exactly the result of
    passing `indices=None` to the forward method.

    We can also pass multi-dimensional `indices` to the forward method, in which case the indices' shape becomes the
    prefix of the result shape: `(*indices.shape, *self.shape)`.
    """

    #: the maximum ID (exclusively)
    max_id: int

    #: the shape of an individual representation
    shape: Tuple[int, ...]

    def __init__(
        self,
        max_id: int,
        shape: Sequence[int],
    ):
        """Initialize the representation module.

        :param max_id:
            The maximum ID (exclusively). Valid Ids reach from 0, ..., max_id-1
        :param shape:
            The shape of an individual representation.
        """
        super().__init__()
        self.max_id = max_id
        self.shape = tuple(shape)

    @abstractmethod
    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get representations for indices.

        :param indices: shape: s
            The indices, or None. If None, this is interpreted as ``torch.arange(self.max_id)`` (although implemented
            more efficiently).

        :return: shape: (``*s``, ``*self.shape``)
            The representations.
        """

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""

    def get_in_canonical_shape(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get representations in canonical shape.

        :param indices: None, shape: (b,) or (b, n)
            The indices. If None, return all representations.

        :return: shape: (b?, n?, d)
            If indices is None, b=1, n=max_id.
            If indices is 1-dimensional, b=indices.shape[0] and n=1.
            If indices is 2-dimensional, b, n = indices.shape
        """
        x = self(indices=indices)
        if indices is None:
            x = x.unsqueeze(dim=0)
        elif indices.ndimension() > 2:
            raise ValueError(
                f"Undefined canonical shape for more than 2-dimensional index tensors: {indices.shape}",
            )
        elif indices.ndimension() == 1:
            x = x.unsqueeze(dim=1)
        return x

    @property
    def embedding_dim(self) -> int:
        """Return the "embedding dimension". Kept for backward compatibility."""
        # TODO: Remove this property and update code to use shape instead
        warnings.warn("The embedding_dim property is deprecated. Use .shape instead.", DeprecationWarning)
        return int(np.prod(self.shape))


class Embedding(RepresentationModule):
    """Trainable embeddings.

    This class provides the same interface as :class:`torch.nn.Embedding` and
    can be used throughout PyKEEN as a more fully featured drop-in replacement.
    """

    normalizer: Optional[Normalizer]
    constrainer: Optional[Constrainer]
    regularizer: Optional[Regularizer]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: Optional[int] = None,
        shape: Union[None, int, Sequence[int]] = None,
        initializer: Hint[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        normalizer: Hint[Normalizer] = None,
        normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Hint[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        regularizer: Optional[Regularizer] = None,
        trainable: bool = True,
    ):
        """Instantiate an embedding with extended functionality.

        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimensionality.
        :param initializer:
            An optional initializer, which takes an uninitialized (num_embeddings, embedding_dim) tensor as input,
            and returns an initialized tensor of same shape and dtype (which may be the same, i.e. the
            initialization may be in-place)
        :param initializer_kwargs:
            Additional keyword arguments passed to the initializer
        :param normalizer:
            A normalization function, which is applied in every forward pass.
        :param normalizer_kwargs:
            Additional keyword arguments passed to the normalizer
        :param constrainer:
            A function which is applied to the weights after each parameter update, without tracking gradients.
            It may be used to enforce model constraints outside of gradient-based training. The function does not need
            to be in-place, but the weight tensor is modified in-place.
        :param constrainer_kwargs:
            Additional keyword arguments passed to the constrainer
        """
        # normalize embedding_dim vs. shape
        _embedding_dim, shape = process_shape(embedding_dim, shape)

        super().__init__(
            max_id=num_embeddings,
            shape=shape,
        )

        self.initializer = cast(Initializer, _handle(
            initializer, initializers, initializer_kwargs, default=nn.init.normal_,
        ))
        self.normalizer = _handle(normalizer, normalizers, normalizer_kwargs)
        self.constrainer = _handle(constrainer, constrainers, constrainer_kwargs)
        self.regularizer = regularizer

        self._embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=_embedding_dim,
        )
        self._embeddings.requires_grad_(trainable)

    @classmethod
    def init_with_device(
        cls,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device,
        initializer: Optional[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        normalizer: Optional[Normalizer] = None,
        normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Optional[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> 'Embedding':  # noqa:E501
        """Create an embedding object on the given device by wrapping :func:`__init__`.

        This method is a hotfix for not being able to pass a device during initialization of
        :class:`torch.nn.Embedding`. Instead the weight is always initialized on CPU and has
        to be moved to GPU afterwards.

        .. seealso::

            https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

        :return:
            The embedding.
        """
        return cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            normalizer=normalizer,
            normalizer_kwargs=normalizer_kwargs,
            constrainer=constrainer,
            constrainer_kwargs=constrainer_kwargs,
        ).to(device=device)

    @property
    def num_embeddings(self) -> int:  # noqa: D401
        """The total number of representations (i.e. the maximum ID)."""
        # wrapper around max_id, for backward compatibility
        return self.max_id

    @property
    def embedding_dim(self) -> int:  # noqa: D401
        """The representation dimension."""
        return self._embeddings.embedding_dim

    def reset_parameters(self) -> None:  # noqa: D102
        # initialize weights in-place
        self._embeddings.weight.data = self.initializer(
            self._embeddings.weight.data.view(self.num_embeddings, *self.shape),
        ).view(self.num_embeddings, self.embedding_dim)

    def post_parameter_update(self):  # noqa: D102
        # apply constraints in-place
        if self.constrainer is not None:
            self._embeddings.weight.data = self.constrainer(self._embeddings.weight.data)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            prefix_shape = (self.max_id,)
            x = self._embeddings.weight
        else:
            prefix_shape = indices.shape
            x = self._embeddings(indices)
        x = x.view(*prefix_shape, *self.shape)
        # verify that contiguity is preserved
        assert x.is_contiguous()
        # TODO: move normalizer / regularizer to base class?
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.regularizer is not None:
            self.regularizer.update(x)
        return x


@dataclass
class EmbeddingSpecification:
    """An embedding specification."""

    embedding_dim: Optional[int] = None
    shape: Union[None, int, Sequence[int]] = None

    initializer: Hint[Initializer] = None
    initializer_kwargs: Optional[Mapping[str, Any]] = None

    normalizer: Hint[Normalizer] = None
    normalizer_kwargs: Optional[Mapping[str, Any]] = None

    constrainer: Hint[Constrainer] = None
    constrainer_kwargs: Optional[Mapping[str, Any]] = None

    regularizer: Optional[Regularizer] = None

    def make(self, *, num_embeddings: int, device: Optional[torch.device] = None) -> Embedding:
        """Create an embedding with this specification."""
        rv = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            shape=self.shape,
            initializer=self.initializer,
            initializer_kwargs=self.initializer_kwargs,
            normalizer=self.normalizer,
            normalizer_kwargs=self.normalizer_kwargs,
            constrainer=self.constrainer,
            constrainer_kwargs=self.constrainer_kwargs,
            regularizer=self.regularizer,
        )
        if device is not None:
            rv = rv.to(device)
        return rv


def process_shape(
    dim: Optional[int],
    shape: Union[None, int, Sequence[int]],
) -> Tuple[int, Sequence[int]]:
    """Make a shape pack."""
    if shape is None and dim is None:
        raise ValueError('Missing both, shape and embedding_dim')
    elif shape is not None and dim is not None:
        raise ValueError('Provided both, shape and embedding_dim')
    elif shape is None and dim is not None:
        shape = (dim,)
    elif isinstance(shape, int) and dim is None:
        dim = shape
        shape = (shape,)
    elif isinstance(shape, Sequence) and dim is None:
        shape = tuple(shape)
        dim = int(np.prod(shape))
    else:
        raise TypeError(f'Invalid type for shape: ({type(shape)}) {shape}')
    return dim, shape


initializers = {
    'xavier_uniform': xavier_normal_,
    'xavier_uniform_norm': compose(
        nn.init.xavier_uniform_,
        functional.normalize,
    ),
    'xavier_normal': xavier_uniform_,
    'xavier_normal_norm': compose(
        nn.init.xavier_normal_,
        functional.normalize,
    ),
    'normal': torch.nn.init.normal_,
    'uniform': torch.nn.init.uniform_,
    'phases': init_phases,
}

constrainers = {
    'normalize': functional.normalize,
    'complex_normalize': complex_normalize,
    'clamp': torch.clamp,
    'clamp_norm': clamp_norm,
}

# TODO add normalization functions
normalizers: Mapping[str, Normalizer] = {}

X = TypeVar('X', bound=Callable)


def _handle(value: Hint[X], lookup: Mapping[str, X], kwargs, default: Optional[X] = None) -> Optional[X]:
    if value is None:
        return default
    elif isinstance(value, str):
        value = lookup[value]
    if kwargs:
        rv = functools.partial(value, **kwargs)  # type: ignore
        return cast(X, rv)
    return value
