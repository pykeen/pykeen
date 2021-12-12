# -*- coding: utf-8 -*-

"""Embedding modules."""

from __future__ import annotations

import functools
import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import torch
import torch.nn
from torch import nn
from torch.nn import functional

from .compositions import CompositionModule, composition_resolver
from .init import (
    init_phases,
    normal_norm_,
    uniform_norm_,
    uniform_norm_p1_,
    xavier_normal_,
    xavier_normal_norm_,
    xavier_uniform_,
    xavier_uniform_norm_,
)
from .utils import TransformerEncoder
from .weighting import EdgeWeighting, SymmetricEdgeWeighting, edge_weight_resolver
from ..constants import AGGREGATIONS
from ..regularizers import Regularizer, regularizer_resolver
from ..triples import CoreTriplesFactory, TriplesFactory
from ..typing import Constrainer, Hint, HintType, Initializer, Normalizer
from ..utils import Bias, activation_resolver, clamp_norm, complex_normalize, convert_to_canonical_shape

__all__ = [
    "RepresentationModule",
    "Embedding",
    "LowRankEmbeddingRepresentation",
    "EmbeddingSpecification",
    "NodePieceRepresentation",
    "CompGCNLayer",
    "CombinedCompGCNRepresentations",
    "SingleCompGCNRepresentation",
    "LabelBasedTransformerRepresentation",
    "SubsetRepresentationModule",
    # Utils
    "constrainers",
    "initializers",
    "normalizers",
]

logger = logging.getLogger(__name__)


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

    def get_in_more_canonical_shape(
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

        Examples:
        >>> emb = EmbeddingSpecification(shape=(20,)).make(num_embeddings=10)
        >>> # Get head representations for given batch indices
        >>> emb.get_in_more_canonical_shape(dim="h", indices=torch.arange(5)).shape
        (5, 1, 1, 1, 20)
        >>> # Get head representations for given 2D batch indices, as e.g. used by fast slcwa scoring
        >>> emb.get_in_more_canonical_shape(dim="h", indices=torch.arange(6).view(2, 3)).shape
        (2, 3, 1, 1, 20)
        >>> # Get head representations for 1:n scoring
        >>> emb.get_in_more_canonical_shape(dim="h", indices=None).shape
        (1, 10, 1, 1, 20)

        :param dim:
            The dimension along which to expand for ``indices=None``, or ``indices.ndimension() == 2``.
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

    @property
    def embedding_dim(self) -> int:
        """Return the "embedding dimension". Kept for backward compatibility."""
        # TODO: Remove this property and update code to use shape instead
        warnings.warn("The embedding_dim property is deprecated. Use .shape instead.", DeprecationWarning)
        return int(np.prod(self.shape))


class SubsetRepresentationModule(RepresentationModule):
    """A representation module, which only exposes a subset of representations of its base."""

    def __init__(
        self,
        base: RepresentationModule,
        max_id: int,
    ):
        """
        Initialize the representations.

        :param base:
            the base representations. have to have a sufficient number of representations, i.e., at least max_id.
        :param max_id:
            the maximum number of relations.
        """
        if max_id > base.max_id:
            raise ValueError(f"Base representations comprise only {base.max_id} representations.")
        super().__init__(max_id, base.shape)
        self.base = base

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            indices = torch.arange(self.max_id)
        return self.base.forward(indices=indices)


class Embedding(RepresentationModule):
    """Trainable embeddings.

    This class provides the same interface as :class:`torch.nn.Embedding` and
    can be used throughout PyKEEN as a more fully featured drop-in replacement.

    It extends it by adding additional options for normalizing, constraining, or applying dropout.

    When a *normalizer* is selected, it is applied in every forward pass. It can be used, e.g., to ensure that the
    embedding vectors are of unit length. A *constrainer* can be used similarly, but it is applied after each parameter
    update (using the post_parameter_update hook), i.e., outside of the automatic gradient computation.

    The optional dropout can also be used as a regularization technique. Moreover, it enables to obtain uncertainty
    estimates via techniques such as `Monte-Carlo dropout <https://arxiv.org/abs/1506.02142>`_. The following simple
    example shows how to obtain different scores for a single triple from an (untrained) model. These scores can be
    considered as samples from a distribution over the scores.

    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()
    >>> from pykeen.nn.emb import EmbeddingSpecification
    >>> spec = EmbeddingSpecification(embedding_dim=3, dropout=0.1)
    >>> from pykeen.models import ERModel
    >>> model = ERModel(
    ...     triples_factory=dataset.training,
    ...     interaction='distmult',
    ...     entity_representations=spec,
    ...     relation_representations=spec,
    ... )
    >>> import torch
    >>> batch = torch.as_tensor(data=[[0, 1, 0]]).repeat(10, 1)
    >>> scores = model.score_hrt(batch)
    """

    normalizer: Optional[Normalizer]
    constrainer: Optional[Constrainer]
    regularizer: Optional[Regularizer]
    dropout: Optional[nn.Dropout]

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
        regularizer: Hint[Regularizer] = None,
        regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        dropout: Optional[float] = None,
    ):
        """Instantiate an embedding with extended functionality.

        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimensionality.
        :param initializer:
            An optional initializer, which takes an uninitialized (num_embeddings, embedding_dim) tensor as input,
            and returns an initialized tensor of same shape and dtype (which may be the same, i.e. the
            initialization may be in-place). Can be passed as a function, or as string corresponding to a key in
            :data:`pykeen.nn.emb.initializers` such as:

            - ``"xavier_uniform"``
            - ``"xavier_uniform_norm"``
            - ``"xavier_normal"``
            - ``"xavier_normal_norm"``
            - ``"normal"``
            - ``"normal_norm"``
            - ``"uniform"``
            - ``"uniform_norm"``
            - ``"init_phases"``
        :param initializer_kwargs:
            Additional keyword arguments passed to the initializer
        :param normalizer:
            A normalization function, which is applied in every forward pass.
        :param normalizer_kwargs:
            Additional keyword arguments passed to the normalizer
        :param constrainer:
            A function which is applied to the weights after each parameter update, without tracking gradients.
            It may be used to enforce model constraints outside of gradient-based training. The function does not need
            to be in-place, but the weight tensor is modified in-place. Can be passed as a function, or as a string
            corresponding to a key in :data:`pykeen.nn.emb.constrainers` such as:

            - ``'normalize'``
            - ``'complex_normalize'``
            - ``'clamp'``
            - ``'clamp_norm'``
        :param constrainer_kwargs:
            Additional keyword arguments passed to the constrainer
        :param regularizer:
            A regularizer, which is applied to the selected embeddings in forward pass
        :param regularizer_kwargs:
            Additional keyword arguments passed to the regularizer
        :param dropout:
            A dropout value for the embeddings.
        """
        # normalize embedding_dim vs. shape
        _embedding_dim, shape = process_shape(embedding_dim, shape)

        if dtype is None:
            dtype = torch.get_default_dtype()

        # work-around until full complex support (torch==1.10 still does not work)
        # TODO: verify that this is our understanding of complex!
        if dtype.is_complex:
            shape = tuple(shape[:-1]) + (2 * shape[-1],)
            _embedding_dim = _embedding_dim * 2

        super().__init__(
            max_id=num_embeddings,
            shape=shape,
        )

        self.initializer = cast(
            Initializer,
            _handle(
                initializer,
                initializers,
                initializer_kwargs,
                default=nn.init.normal_,
                label="initializer",
            ),
        )
        self.normalizer = _handle(normalizer, normalizers, normalizer_kwargs, label="normalizer")
        self.constrainer = _handle(constrainer, constrainers, constrainer_kwargs, label="constrainer")
        if regularizer is not None:
            regularizer = regularizer_resolver.make(regularizer, pos_kwargs=regularizer_kwargs)
        self.regularizer = regularizer

        self._embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=_embedding_dim,
        )
        self._embeddings.requires_grad_(trainable)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

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
    ) -> "Embedding":  # noqa:E501
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
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class LowRankEmbeddingRepresentation(RepresentationModule):
    r"""
    Low-rank embedding factorization.

    This representation reduces the number of trainable parameters by not learning independent weights for each index,
    but rather having shared bases among all indices, and only learn the weights of the linear combination.

    .. math ::
        E[i] = \sum_k B[i, k] * W[k]
    """

    def __init__(
        self,
        *,
        max_id: int,
        shape: Sequence[int],
        num_bases: int = 3,
        weight_initializer: Initializer = uniform_norm_p1_,
        **kwargs,
    ):
        """
        Initialize the representations.

        :param max_id:
            the maximum ID (exclusively). Valid Ids reach from 0, ..., max_id-1
        :param shape:
            the shape of an individual base representation.
        :param num_bases:
            the number of bases. More bases increase expressivity, but also increase the number of trainable parameters.
        :param weight_initializer:
            the initializer for basis weights
        :param kwargs:
            additional keyword based arguments passed to :class:`pykeen.nn.emb.Embedding`, which is used for the base
            representations.
        """
        super().__init__(max_id=max_id, shape=shape)
        self.bases = Embedding(num_embeddings=num_bases, shape=shape, **kwargs)
        self.weight_initializer = weight_initializer
        self.weight = nn.Parameter(torch.empty(max_id, num_bases))
        self.reset_parameters()

    def reset_parameters(self) -> None:  # noqa: D102
        self.bases.reset_parameters()
        self.weight.data = self.weight_initializer(self.weight)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # get all base representations, shape: (num_bases, *shape)
        bases = self.bases(indices=None)
        # get base weights, shape: (*batch_dims, num_bases)
        weight = self.weight
        if indices is not None:
            weight = weight[indices]
        # weighted linear combination of bases, shape: (*batch_dims, *shape)
        return torch.tensordot(weight, bases, dims=([-1], [0]))


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

    regularizer: Hint[Regularizer] = None
    regularizer_kwargs: Optional[Mapping[str, Any]] = None

    dtype: Optional[torch.dtype] = None
    dropout: Optional[float] = None

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
            regularizer_kwargs=self.regularizer_kwargs,
            dtype=self.dtype,
            dropout=self.dropout,
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
        raise ValueError("Missing both, shape and embedding_dim")
    elif shape is not None and dim is not None:
        raise ValueError("Provided both, shape and embedding_dim")
    elif shape is None and dim is not None:
        shape = (dim,)
    elif isinstance(shape, int) and dim is None:
        dim = shape
        shape = (shape,)
    elif isinstance(shape, Sequence) and dim is None:
        shape = tuple(shape)
        dim = int(np.prod(shape))
    else:
        raise TypeError(f"Invalid type for shape: ({type(shape)}) {shape}")
    return dim, shape


#: Initializers
initializers = {
    "xavier_uniform": xavier_uniform_,
    "xavier_uniform_norm": xavier_uniform_norm_,
    "xavier_normal": xavier_normal_,
    "xavier_normal_norm": xavier_normal_norm_,
    "normal": torch.nn.init.normal_,
    "normal_norm": normal_norm_,
    "uniform": torch.nn.init.uniform_,
    "uniform_norm": uniform_norm_,
    "phases": init_phases,
    "init_phases": init_phases,
}

#: Constrainers
constrainers = {
    "normalize": functional.normalize,
    "complex_normalize": complex_normalize,
    "clamp": torch.clamp,
    "clamp_norm": clamp_norm,
}

# TODO add normalization functions
normalizers: Mapping[str, Normalizer] = {}

X = TypeVar("X", bound=Callable)


def _handle(
    value: Hint[X],
    lookup: Mapping[str, X],
    kwargs,
    default: Optional[X] = None,
    label: Optional[str] = None,
) -> Optional[X]:
    if value is None:
        return default
    elif isinstance(value, str):
        try:
            value = lookup[value]
        except KeyError:
            raise KeyError(f"{value} is an invalid {label}. Try one of: {sorted(lookup)}")
    if kwargs:
        rv = functools.partial(value, **kwargs)  # type: ignore
        return cast(X, rv)
    return value


class CompGCNLayer(nn.Module):
    """A single layer of the CompGCN model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_relation_bias: bool = False,
        composition: Hint[CompositionModule] = None,
        activation: Hint[nn.Module] = nn.Identity,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        edge_weighting: HintType[EdgeWeighting] = SymmetricEdgeWeighting,
    ):
        """
        Initialize the module.

        :param input_dim:
            The input dimension.
        :param output_dim:
            The output dimension. If None, equals the input dimension.
        :param dropout:
            The dropout to use for forward and backward edges.
        :param use_bias:  # TODO: do we really need this? it comes before a mandatory batch norm layer
            Whether to use bias.
        :param use_relation_bias:
            Whether to use a bias for the relation transformation.
        :param composition:
            The composition function.
        :param activation:
            The activation to use.
        :param activation_kwargs:
            Additional key-word based arguments passed to the activation.
        """
        super().__init__()

        # normalize output dimension
        output_dim = output_dim or input_dim

        # entity-relation composition
        self.composition = composition_resolver.make(composition)

        # edge weighting
        self.edge_weighting: EdgeWeighting = edge_weight_resolver.make(edge_weighting)

        # message passing weights
        self.w_loop = nn.Parameter(data=torch.empty(input_dim, output_dim))
        self.w_fwd = nn.Parameter(data=torch.empty(input_dim, output_dim))
        self.w_bwd = nn.Parameter(data=torch.empty(input_dim, output_dim))

        # linear relation transformation
        self.w_rel = nn.Linear(in_features=input_dim, out_features=output_dim, bias=use_relation_bias)

        # layer-specific self-loop relation representation
        self.self_loop = nn.Parameter(data=torch.empty(1, input_dim))

        # other components
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        self.bias = Bias(output_dim) if use_bias else None
        self.activation = activation_resolver.make(query=activation, pos_kwargs=activation_kwargs)

        # initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the model's parameters."""
        for w in (
            self.w_loop,
            self.w_fwd,
            self.w_bwd,
            self.self_loop,
        ):
            nn.init.xavier_uniform_(w)
        self.bias.reset_parameters()
        self.w_rel.reset_parameters()

    def message(
        self,
        x_e: torch.FloatTensor,
        x_r: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        weight: nn.Parameter,
    ) -> torch.FloatTensor:
        """
        Perform message passing.

        :param x_e: shape: (num_entities, input_dim)
            The entity representations.
        :param x_r: shape: (2 * num_relations, input_dim)
            The relation representations (including inverse relations).
        :param edge_index: shape: (2, num_edges)
            The edge index, pairs of source and target entity for each triple.
        :param edge_type: shape (num_edges,)
            The edge type, i.e., relation ID, for each triple.
        :param weight:
            The transformation weight.

        :return:
            The updated entity representations.
        """
        # split
        source, target = edge_index

        # compose
        m = self.composition(x_e[source], x_r[edge_type])

        # transform
        m = m @ weight

        # normalization
        m = m * self.edge_weighting(source=source, target=target).unsqueeze(dim=-1)

        # aggregate by sum
        x_e = x_e.new_zeros(x_e.shape[0], m.shape[1]).index_add(dim=0, index=target, source=m)

        # dropout
        x_e = self.drop(x_e)

        return x_e

    def forward(
        self,
        x_e: torch.FloatTensor,
        x_r: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""
        Update entity and relation representations.

        .. math ::
            X_E'[e] = \frac{1}{3} \left(
                X_E W_s
                + \left( \sum_{h,r,e \in T} \alpha(h, e) \phi(X_E[h], X_R[r]) W_f \right)
                + \left( \sum_{e,r,t \in T} \alpha(e, t) \phi(X_E[t], X_R[r^{-1}]) W_b \right)
            \right)

        :param x_e: shape: (num_entities, input_dim)
            The entity representations.
        :param x_r: shape: (2 * num_relations, input_dim)
            The relation representations (including inverse relations).
        :param edge_index: shape: (2, num_edges)
            The edge index, pairs of source and target entity for each triple.
        :param edge_type: shape (num_edges,)
            The edge type, i.e., relation ID, for each triple.

        :return: shape: (num_entities, output_dim) / (2 * num_relations, output_dim)
            The updated entity and relation representations.
        """
        # prepare for inverse relations
        edge_type = 2 * edge_type
        # update entity representations: mean over self-loops / forward edges / backward edges
        x_e = (
            self.composition(x_e, self.self_loop) @ self.w_loop
            + self.message(x_e=x_e, x_r=x_r, edge_index=edge_index, edge_type=edge_type, weight=self.w_fwd)
            + self.message(x_e=x_e, x_r=x_r, edge_index=edge_index.flip(0), edge_type=edge_type + 1, weight=self.w_bwd)
        ) / 3

        if self.bias:
            x_e = self.bias(x_e)
        x_e = self.bn(x_e)
        x_e = self.activation(x_e)

        # Relation transformation
        x_r = self.w_rel(x_r)
        return x_e, x_r


class CombinedCompGCNRepresentations(nn.Module):
    """A sequence of CompGCN layers."""

    # Buffered enriched entity and relation representations
    enriched_representations: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        embedding_specification: EmbeddingSpecification,
        num_layers: Optional[int] = 1,
        dims: Union[None, int, Sequence[int]] = None,
        layer_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the combined entity and relation representation module.

        :param triples_factory:
            The triples factory containing the training triples.
        :param embedding_specification:
            An embedding specification for the base entity and relation representations.
        :param num_layers:
            The number of message passing layers to use. If None, will be inferred by len(dims), i.e., requires dims to
            be a sequence / list.
        :param dims:
            The hidden dimensions to use. If None, defaults to the embedding dimension of the base representations.
            If an integer, is the same for all layers. The last dimension is equal to the output dimension.
        :param layer_kwargs:
            Additional key-word based parameters passed to the individual layers; cf. CompGCNLayer.
        """
        super().__init__()
        # TODO: Check
        assert triples_factory.create_inverse_triples
        self.entity_representations = embedding_specification.make(
            num_embeddings=triples_factory.num_entities,
        )
        self.relation_representations = embedding_specification.make(
            num_embeddings=2 * triples_factory.real_num_relations,
        )
        input_dim = self.entity_representations.embedding_dim
        assert self.relation_representations.embedding_dim == input_dim

        # hidden dimension normalization
        if dims is None:
            dims = input_dim
        if isinstance(dims, int):
            if num_layers is None:
                raise ValueError
            else:
                dims = [dims] * num_layers
        if len(dims) != num_layers:
            raise ValueError(
                f"The number of provided dimensions ({len(dims)}) must equal the number of layers ({num_layers}).",
            )
        self.output_dim = dims[-1]

        # Create message passing layers
        layers = []
        for input_dim, output_dim in zip(itertools.chain([input_dim], dims), dims):
            layers.append(
                CompGCNLayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    **(layer_kwargs or {}),
                )
            )
        self.layers = nn.ModuleList(layers)

        # register buffers for adjacency matrix; we use the same format as PyTorch Geometric
        # TODO: This always uses all training triples for message passing
        self.register_buffer(name="edge_index", tensor=triples_factory.mapped_triples[:, [0, 2]].t())
        self.register_buffer(name="edge_type", tensor=triples_factory.mapped_triples[:, 1])

        # initialize buffer of enriched representations
        self.enriched_representations = None

    def post_parameter_update(self) -> None:  # noqa: D102
        # invalidate enriched embeddings
        self.enriched_representations = None

    def train(self, mode: bool = True):  # noqa: D102
        # when changing from evaluation to training mode, the buffered representations have been computed without
        # gradient tracking. hence, we need to invalidate them.
        # note: this occurs in practice when continuing training after evaluation.
        if mode and not self.training:
            self.enriched_representations = None
        return super().train(mode=mode)

    def forward(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute enriched representations."""
        if self.enriched_representations is None:
            x_e = self.entity_representations()
            x_r = self.relation_representations()
            # enrich
            for layer in self.layers:
                x_e, x_r = layer(x_e=x_e, x_r=x_r, edge_index=self.edge_index, edge_type=self.edge_type)
            self.enriched_representations = (x_e, x_r)
        return self.enriched_representations

    def split(self) -> Tuple["SingleCompGCNRepresentation", "SingleCompGCNRepresentation"]:
        """Return the separated representations."""
        return (
            SingleCompGCNRepresentation(self, position=0),
            SingleCompGCNRepresentation(self, position=1),
        )


class SingleCompGCNRepresentation(RepresentationModule):
    """A wrapper around the combined representation module."""

    def __init__(
        self,
        combined: CombinedCompGCNRepresentations,
        position: int = 0,
    ):
        """
        Initialize the module.

        :param combined:
            The combined representations.
        :param position:
            The position, either 0 for entities, or 1 for relations.
        """
        if position == 0:  # entity
            max_id = combined.entity_representations.max_id
            shape = (combined.output_dim,)
        elif position == 1:  # relation
            max_id = combined.relation_representations.max_id
            shape = (combined.output_dim,)
        else:
            raise ValueError
        super().__init__(max_id=max_id, shape=shape)
        self.combined = combined
        self.position = position
        self.reset_parameters()

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        x = self.combined()[self.position]
        if indices is not None:
            x = x[indices]
        return x


def _sample(rs: torch.LongTensor, k: int) -> torch.LongTensor:
    """Sample without replacement."""
    return rs[torch.randperm(rs.shape[0])[:k]]


def tokenize(
    triples_factory: CoreTriplesFactory,
    num_tokens: int,
) -> torch.LongTensor:
    """
    Tokenize entities by representing them as a bag of relations.

    :param triples_factory:
        the triples factory containing the ID-based triples.
    :param num_tokens:
        the number of relation IDs to select for each entity

    :return: shape: (num_entities, num_tokens), -1 <= res < 2 * num_relations
        the selected relation IDs for each entity. -1 is used as a padding token.
    """
    mapped_triples = triples_factory.mapped_triples
    if triples_factory.create_inverse_triples:
        # inverse triples are created afterwards implicitly
        mapped_triples = mapped_triples[mapped_triples[:, 1] < triples_factory.real_num_relations]

    # tokenize: represent entities by bag of relations
    h, r, t = mapped_triples.t()

    # collect candidates
    e2r = defaultdict(set)
    for e, r in (
        torch.cat(
            [
                torch.stack([h, r], dim=1),
                torch.stack([t, r + triples_factory.real_num_relations], dim=1),
            ],
            dim=0,
        )
        .unique(dim=0)
        .tolist()
    ):
        e2r[e].add(r)

    # randomly sample without replacement num_tokens relations for each entity
    assignment = torch.full(
        size=(triples_factory.num_entities, num_tokens),
        dtype=torch.long,
        fill_value=2 * triples_factory.real_num_relations,
    )
    for e, rs in e2r.items():
        rs = torch.as_tensor(data=list(rs), dtype=torch.long)
        rs = _sample(rs=rs, k=num_tokens)
        assignment[e, : len(rs)] = rs

    return assignment


def resolve_aggregation(
    aggregation: Union[None, str, Callable[[torch.FloatTensor, int], torch.FloatTensor]],
) -> Callable[[torch.FloatTensor, int], torch.FloatTensor]:
    """
    Resolve the aggregation function.

    .. warning ::
        This function does *not* check whether torch.<aggregation> is a method which is a valid aggregation.

    :param aggregation:
        the aggregation choice. Can be either
        1. None, in which case the torch.mean is returned
        2. a string, in which case torch.<aggregation> is returned
        3. a callable, which is returned without change

    :return:
        the chosen aggregation function.
    """
    if aggregation is None:
        return torch.mean

    if isinstance(aggregation, str):
        if aggregation not in AGGREGATIONS:
            logger.warning(
                f"aggregation={aggregation} is not one of the predefined ones ({sorted(AGGREGATIONS.keys())}).",
            )
        return getattr(torch, aggregation)

    return aggregation


class NodePieceRepresentation(RepresentationModule):
    r"""
    Basic implementation of node piece decomposition [galkin2021]_.

    .. math ::
        x_e = agg(\{T[t] \mid t \in tokens(e) \})

    where $T$ are token representations, $tokens$ selects a fixed number of $k$ tokens for each entity, and $agg$ is
    an aggregation function, which aggregates the individual token representations to a single entity representation.

    .. note ::
        This implementation currently only supports representation of entities by bag-of-relations.
    """

    #: the token representations
    tokens: RepresentationModule

    #: the entity-to-token mapping
    assignment: torch.LongTensor

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        token_representation: Union[EmbeddingSpecification, RepresentationModule],
        aggregation: Union[None, str, Callable[[torch.FloatTensor, int], torch.FloatTensor]] = None,
        num_tokens: int = 2,
        shape: Optional[Sequence[int]] = None,
    ):
        """
        Initialize the representation.

        :param triples_factory:
            the triples factory
        :param token_representation:
            the token representation specification, or pre-instantiated representation module. For the latter, the
            number of representations must be $2 * num_relations + 1$.
        :param aggregation:
            aggregation of multiple token representations to a single entity representation. By default,
            this uses :func:`torch.mean`. If a string is provided, the module assumes that this refers to a top-level
            torch function, e.g. "mean" for :func:`torch.mean`, or "sum" for func:`torch.sum`. An aggregation can
            also have trainable parameters, .e.g., ``MLP(mean(MLP(tokens)))`` (cf. DeepSets from [zaheer2017]_). In
            this case, the module has to be created outside of this component.

            We could also have aggregations which result in differently shapes output, e.g. a concatenation of all
            token embeddings resulting in shape ``(num_tokens * d,)``. In this case, `shape` must be provided.

            The aggregation takes two arguments: the (batched) tensor of token representations, in shape
            ``(*, num_tokens, *dt)``, and the index along which to aggregate.
        :param num_tokens:
            the number of tokens for each entity.
        :param shape:
            the shape of an individual representation. Only necessary, if aggregation results in a change of dimensions.
        """
        # create token representations
        # normal relations + inverse relations + padding
        total_num_tokens = 2 * triples_factory.real_num_relations + 1
        if isinstance(token_representation, EmbeddingSpecification):
            token_representation = token_representation.make(
                num_embeddings=total_num_tokens,
            )
        if token_representation.max_id != total_num_tokens:
            raise ValueError(
                f"If a pre-instantiated representation is provided, it has to have 2 * num_relations + 1= "
                f"{total_num_tokens} representations, but has {token_representation.max_id}",
            )

        # super init; has to happen *before* any parameter or buffer is assigned
        super().__init__(max_id=triples_factory.num_entities, shape=shape or token_representation.shape)

        # Assign default aggregation
        self.aggregation = resolve_aggregation(aggregation=aggregation)

        # assign module
        self.tokens = token_representation
        self.aggregation_index = -(1 + len(token_representation.shape))
        self.register_buffer(
            name="assignment",
            tensor=tokenize(triples_factory=triples_factory, num_tokens=num_tokens),
        )

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # get token IDs, shape: (*, k)
        token_ids = self.assignment
        if indices is not None:
            token_ids = token_ids[indices]

        # lookup token representations, shape: (*, k, d)
        x = self.tokens(token_ids)

        # aggregate
        x = self.aggregation(x, self.aggregation_index)

        return x


class LabelBasedTransformerRepresentation(RepresentationModule):
    """
    Label-based representations using a transformer encoder.

    Example Usage:

    Entity representations are obtained by encoding the labels with a Transformer model. The transformer
    model becomes part of the KGE model, and its parameters are trained jointly.

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.nn.emb import EmbeddingSpecification, LabelBasedTransformerRepresentation
        from pykeen.models import ERModel

        dataset = get_dataset(dataset="nations")
        entity_representations = LabelBasedTransformerRepresentation.from_triples_factory(
            triples_factory=dataset.training,
        )
        model = ERModel(
            interaction="ermlp",
            entity_representations=entity_representations,
            relation_representations=EmbeddingSpecification(shape=entity_representations.shape),
        )
    """

    def __init__(
        self,
        labels: Sequence[str],
        pretrained_model_name_or_path: str = "bert-base-cased",
        max_length: int = 512,
    ):
        """
        Initialize the representation.

        :param labels:
            the labels
        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. AutoModel.from_pretrained
        :param max_length: >0
            the maximum number of tokens to pad/trim the labels to
        """
        encoder = TransformerEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_length=max_length,
        )
        # infer shape
        shape = encoder.encode_all(labels[0:1]).shape[1:]
        super().__init__(max_id=len(labels), shape=shape)

        self.labels = labels
        # assign after super, since they should be properly registered as submodules
        self.encoder = encoder

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "LabelBasedTransformerRepresentation":
        """
        Prepare a label-based transformer representations with labels from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :func:`LabelBasedTransformerRepresentation.__init__`

        :raise ImportError:
            if the transformers library could not be imported
        """
        id_to_label = triples_factory.entity_id_to_label if for_entities else triples_factory.relation_id_to_label
        return cls(
            labels=[id_to_label[i] for i in range(len(id_to_label))],
            **kwargs,
        )

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            indices = torch.arange(self.max_id)
        uniq, inverse = indices.unique(return_inverse=True)
        x = self.encoder(
            labels=[self.labels[i] for i in uniq.tolist()],
        )
        return x[inverse]
