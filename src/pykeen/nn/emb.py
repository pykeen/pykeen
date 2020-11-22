# -*- coding: utf-8 -*-

"""Embedding modules."""

import dataclasses
import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence, Union

import numpy
import torch
import torch.nn
from torch import nn

from ..regularizers import Regularizer
from ..typing import Constrainer, Initializer, Normalizer

__all__ = [
    'RepresentationModule',
    'Embedding',
    'EmbeddingSpecification',
    'LiteralRepresentations',
]

logger = logging.getLogger(__name__)


class RepresentationModule(nn.Module, ABC):
    """A base class for obtaining representations for entities/relations."""

    #: The shape of a single representation
    shape: Sequence[int]

    #: The maximum admissible ID (excl.)
    max_id: int

    def __init__(self, shape: Sequence[int], max_id: int):
        super().__init__()
        self.shape = shape
        self.max_id = max_id

    @abstractmethod
    def forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """Get representations for indices.

        :param indices: shape: (m,)
            The indices, or None. If None, return all representations.

        :return: shape: (m, d)
            The representations.
        """
        raise NotImplementedError

    def get_in_canonical_shape(
        self,
        indices: Optional[torch.LongTensor] = None,
        reshape_dim: Optional[Sequence[int]] = None,
    ) -> torch.FloatTensor:
        """Get representations in canonical shape.

        The canonical shape is given by (batch_size, 1, ``*``) if indices is not None, where batch_size=len(indices),
        or (1, num, ``*``) if indices is None with num equal to the total number of embeddings.


        :param indices:
            The indices. If None, return all embeddings.
        :param reshape_dim:
            Optionally reshape the last dimension.

        :return: shape: (batch_size, num_embeddings, ``*``)
        """
        x = self(indices=indices)
        if indices is None:
            x = x.unsqueeze(dim=0)
        else:
            x = x.unsqueeze(dim=1)
        if reshape_dim is not None:
            x = x.view(*x.shape[:-1], *reshape_dim)
        return x

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""


@dataclasses.dataclass
class EmbeddingSpecification:
    """An embedding specification."""

    # embedding_dim: int
    # shape: Optional[Sequence[int]] = None

    initializer: Optional[Initializer] = None
    initializer_kwargs: Optional[Mapping[str, Any]] = None

    normalizer: Optional[Normalizer] = None
    normalizer_kwargs: Optional[Mapping[str, Any]] = None

    constrainer: Optional[Constrainer] = None
    constrainer_kwargs: Optional[Mapping[str, Any]] = None

    regularizer: Optional[Regularizer] = None

    def make(
        self,
        num_embeddings: int,
        embedding_dim: Optional[int],
        shape: Optional[Union[int, Sequence[int]]],
    ) -> 'Embedding':
        """Create an embedding with this specification."""
        return Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            shape=shape,
            initializer=self.initializer,
            initializer_kwargs=self.initializer_kwargs,
            normalizer=self.normalizer,
            normalizer_kwargs=self.normalizer_kwargs,
            constrainer=self.constrainer,
            constrainer_kwargs=self.constrainer_kwargs,
            regularizer=self.regularizer,
        )


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
        initializer: Optional[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        normalizer: Optional[Normalizer] = None,
        normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Optional[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        regularizer: Optional[Regularizer] = None,
    ):
        """Instantiate an embedding with extended functionality.

        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimensionality.
        :param shape:
            The embedding shape. If given, shape supersedes embedding_dim, with setting embedding_dim = prod(shape).
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

        :raises ValueError:
            If neither shape nor embedding_dim are given.
        """
        if shape is None and embedding_dim is None:
            raise ValueError('Missing both shape and embedding_dim')
        elif shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            else:
                shape = shape
            embedding_dim = numpy.prod(shape)
        else:
            assert embedding_dim is not None
            shape = (embedding_dim,)
        super().__init__(shape=shape, max_id=num_embeddings)

        if initializer is None:
            initializer = nn.init.normal_

        if initializer_kwargs:
            initializer = functools.partial(initializer, **initializer_kwargs)
        self.initializer = initializer

        if constrainer is not None and constrainer_kwargs:
            constrainer = functools.partial(constrainer, **constrainer_kwargs)
        self.constrainer = constrainer

        # TODO: Move regularizer and normalizer to RepresentationModule?
        if normalizer is not None and normalizer_kwargs:
            normalizer = functools.partial(normalizer, **normalizer_kwargs)
        self.normalizer = normalizer

        self.regularizer = regularizer

        self._embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

    @classmethod
    def from_specification(
        cls,
        num_embeddings: int,
        embedding_dim: Optional[int] = None,
        shape: Optional[Union[int, Sequence[int]]] = None,
        specification: Optional[EmbeddingSpecification] = None,
    ) -> 'Embedding':
        """Create an embedding based on a specification.

        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimension.
        :param shape:
            The embedding shape. If given, shape supersedes embedding_dim, with setting embedding_dim = prod(shape).
        :param specification:
            The specification.
        :return:
            An embedding object.
        """
        if specification is None:
            specification = EmbeddingSpecification()
        return specification.make(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            shape=shape,
        )

    @property
    def num_embeddings(self) -> int:  # noqa: D401
        """The total number of representations (i.e. the maximum ID)."""
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
            x = self._embeddings.weight
        else:
            x = self._embeddings(indices)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.regularizer is not None:
            self.regularizer.update(x)
        return x

    def get_in_canonical_shape(
        self,
        indices: Optional[torch.LongTensor] = None,
        reshape_dim: Optional[Sequence[int]] = None,
    ) -> torch.FloatTensor:
        """Get embedding in canonical shape.

        :param indices:
            The indices. If None, return all embeddings.
        :param reshape_dim:
            Optionally reshape the last dimension.

        :return: shape: (batch_size, num_embeddings, d)
        """
        if len(self.shape) > 1 and reshape_dim is None:
            reshape_dim = self.shape
        return super().get_in_canonical_shape(
            indices=indices,
            reshape_dim=reshape_dim,
        )


class LiteralRepresentations(Embedding):
    """Literal representations."""

    def __init__(
        self,
        numeric_literals: torch.FloatTensor,
    ):
        num_embeddings, embedding_dim = numeric_literals.shape
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            initializer=lambda x: numeric_literals,  # initialize with the literals
        )
        # freeze
        self._embeddings.requires_grad_(False)
