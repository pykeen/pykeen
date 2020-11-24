# -*- coding: utf-8 -*-

"""Embedding modules."""

import dataclasses
import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy
import torch
import torch.nn
from torch import nn

from ..regularizers import Regularizer
from ..typing import Constrainer, Initializer, Normalizer
from ..utils import upgrade_to_sequence

__all__ = [
    'RepresentationModule',
    'Embedding',
    'EmbeddingSpecification',
    'LiteralRepresentations',
]

logger = logging.getLogger(__name__)

HEAD_DIM = 1
RELATION_DIM = 2
TAIL_DIM = 3
DIMS = dict(h=HEAD_DIM, r=RELATION_DIM, t=TAIL_DIM)


def _normalize_dim(dim: Union[int, str]) -> int:
    """Normalize the dimension selection."""
    if isinstance(dim, int):
        return dim
    return DIMS[dim.lower()[0]]


def get_expected_canonical_shape(
    indices_shape: Union[None, int, Tuple[int, int]],
    dim: Union[str, int],
    suffix_shape: Union[int, Sequence[int]],
    num: Optional[int] = None,
) -> Tuple[int, ...]:
    """
    Calculate the expected canonical shape for the given parameters.

    :param indices_shape:
        The shape of the indices, or None, if no indices are to be provided.
    :param dim:
        The dimension, either symbolic, or numeric.
    :param suffix_shape:
        The suffix-shape.
    :param num:
        The number of representations, if indices_shape is None, i.e. 1-n scoring.

    :return: (batch_size, num_heads, num_relations, num_tails, ``*``).
        The expected shape, a tuple of at least 5 positive integers.
    """
    exp_shape = [1, 1, 1, 1] + list(suffix_shape)
    dim = _normalize_dim(dim=dim)
    if indices_shape is None:  # 1-n scoring
        exp_shape[dim] = num
    else:  # batch dimension
        exp_shape[0] = indices_shape[0]
        if len(indices_shape) > 1:  # multi-target batching
            exp_shape[dim] = indices_shape[1]
    return tuple(exp_shape)


def convert_to_canonical_shape(
    x: torch.FloatTensor,
    dim: Union[int, str],
    num: Optional[int] = None,
    batch_size: int = 1,
    suffix_shape: Union[int, Sequence[int]] = -1,
) -> torch.FloatTensor:
    """
    Convert a tensor to canonical shape.

    :param x:
        The tensor in compatible shape.
    :param dim:
        The "num" dimension.
    :param batch_size:
        The batch size.
    :param num:
        The number.
    :param suffix_shape:
        The suffix shape.

    :return: shape: (batch_size, num_heads, num_relations, num_tails, ``*``)
        A tensor in canonical shape.
    """
    if num is None:
        num = x.shape[0]
    suffix_shape = upgrade_to_sequence(suffix_shape)
    shape = [batch_size, 1, 1, 1]
    dim = _normalize_dim(dim=dim)
    shape[dim] = num
    return x.view(*shape, *suffix_shape)


class RepresentationModule(nn.Module, ABC):
    """A base class for obtaining representations for entities/relations."""

    #: The shape of a single representation
    shape: Sequence[int]

    #: The maximum admissible ID (excl.)
    max_id: int

    def __init__(self, shape: Iterable[int], max_id: int):
        super().__init__()
        self.shape = tuple(shape)
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
        dim: Union[int, str],
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get representations in canonical shape.

        The canonical shape is given as

        (batch_size, d_1, d_2, d_3, ``*``)

        fulfilling the following properties:

        Let i = dim. If indices is None, the return shape is (1, d_1, d_2, d_3) with d_i = num_representations,
        d_i = 1 else. If indices is not None, then batch_size = indices.shape[0], and d_i = 1 if
        indices.ndimension() = 1 else d_i = indices.shape[1]

        The canonical shape is given by (batch_size, 1, ``*``) if indices is not None, where batch_size=len(indices),
        or (1, num, ``*``) if indices is None with num equal to the total number of embeddings.


        :param dim:
            The dimension along which to expand for indices = None, or indices.ndimension() == 2.
        :param indices:
            The indices. Either None, in which care all embeddings are returned, or a 1 or 2 dimensional index tensor.

        :return: shape: (batch_size, d1, d2, d3, ``*self.shape``)
        """
        r_shape: Tuple[int, ...]
        if indices is None:
            x = self(indices=indices)
            r_shape = (1, self.max_id)
        else:
            flat_indices = indices.view(-1)
            x = self(indices=flat_indices)
            if indices.ndimension() > 1:
                x = x.view(*indices.shape, -1)
            r_shape = tuple(indices.shape)
            if len(r_shape) < 2:
                r_shape = r_shape + (1,)
        return convert_to_canonical_shape(x=x, dim=dim, num=r_shape[1], batch_size=r_shape[0], suffix_shape=self.shape)

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""


@dataclasses.dataclass
class EmbeddingSpecification:
    """An embedding specification."""

    embedding_dim: Optional[int] = None
    shape: Optional[Sequence[int]] = None

    dtype: Optional[torch.dtype] = None

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
    ) -> 'Embedding':
        """Create an embedding with this specification."""
        return Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            shape=self.shape,
            dtype=self.dtype,
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
        dtype: Optional[torch.dtype] = None,
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
            raise ValueError('Missing both, shape and embedding_dim')
        elif shape is not None and embedding_dim is not None:
            raise ValueError('Provided both, shape and embedding_dim')
        elif shape is None and embedding_dim is not None:
            shape = (embedding_dim,)
        elif isinstance(shape, int) and embedding_dim is None:
            embedding_dim = shape
            shape = (shape,)
        elif isinstance(shape, Sequence) and embedding_dim is None:
            shape = tuple(shape)
            embedding_dim = int(numpy.prod(shape))
        else:
            raise TypeError(f'Invalid type for shape: ({type(shape)}) {shape}')

        assert isinstance(shape, tuple)
        assert isinstance(embedding_dim, int)

        if dtype is None:
            dtype = torch.get_default_dtype()

        # work-around until full complex support
        # TODO: verify that this is our understanding of complex!
        if dtype.is_complex:
            shape = shape[:-1] + (2 * shape[-1],)
            embedding_dim = embedding_dim * 2
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
        specification: Optional[EmbeddingSpecification] = None,
    ) -> 'Embedding':
        """Create an embedding based on a specification.

        :param num_embeddings: >0
            The number of embeddings.
        :param specification:
            The specification.
        :return:
            An embedding object.
        """
        if specification is None:
            specification = EmbeddingSpecification()
        return specification.make(
            num_embeddings=num_embeddings,
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
        x = x.view(x.shape[0], *self.shape)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.regularizer is not None:
            self.regularizer.update(x)
        return x


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
