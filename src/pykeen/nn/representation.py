# -*- coding: utf-8 -*-

"""Representation modules."""

from __future__ import annotations

import itertools
import logging
import math
import string
import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import more_itertools
import numpy
import numpy as np
import torch
import torch.nn
from class_resolver import FunctionResolver, HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn
from torch.nn import functional

from .combination import Combination, combination_resolver
from .compositions import CompositionModule, composition_resolver
from .init import initializer_resolver, uniform_norm_p1_
from .text import TextEncoder, text_encoder_resolver
from .utils import PyOBOCache, ShapeError, TextCache, WikidataCache
from .weighting import EdgeWeighting, SymmetricEdgeWeighting, edge_weight_resolver
from ..datasets import Dataset
from ..regularizers import Regularizer, regularizer_resolver
from ..triples import CoreTriplesFactory, TriplesFactory
from ..triples.triples_factory import Labeling
from ..typing import Constrainer, Hint, HintType, Initializer, Normalizer, OneOrSequence
from ..utils import (
    Bias,
    ExtraReprMixin,
    broadcast_upgrade_to_sequences,
    clamp_norm,
    complex_normalize,
    einsum,
    get_edge_index,
    get_preferred_device,
    upgrade_to_sequence,
)

__all__ = [
    "Representation",
    "Embedding",
    "LowRankRepresentation",
    "CompGCNLayer",
    "CombinedCompGCNRepresentations",
    "PartitionRepresentation",
    "BackfillRepresentation",
    "SingleCompGCNRepresentation",
    "SubsetRepresentation",
    "CombinedRepresentation",
    "TensorTrainRepresentation",
    "TransformedRepresentation",
    "TextRepresentation",
    "CachedTextRepresentation",
    "WikidataTextRepresentation",
    "BiomedicalCURIERepresentation",
    # Utils
    "constrainer_resolver",
    "normalizer_resolver",
]

logger = logging.getLogger(__name__)


class Representation(nn.Module, ExtraReprMixin, ABC):
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

    #: a normalizer for individual representations
    normalizer: Optional[Normalizer]

    #: a regularizer for individual representations
    regularizer: Optional[Regularizer]

    #: dropout
    dropout: Optional[nn.Dropout]

    def __init__(
        self,
        max_id: int,
        shape: OneOrSequence[int] = 64,
        normalizer: HintOrType[Normalizer] = None,
        normalizer_kwargs: OptionalKwargs = None,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        dropout: Optional[float] = None,
        unique: Optional[bool] = None,
    ):
        """Initialize the representation module.

        :param max_id:
            The maximum ID (exclusively). Valid Ids reach from 0, ..., max_id-1
        :param shape:
            The shape of an individual representation.
        :param normalizer:
            A normalization function, which is applied to the selected representations in every forward pass.
        :param normalizer_kwargs:
            Additional keyword arguments passed to the normalizer
        :param regularizer:
            An output regularizer, which is applied to the selected representations in forward pass
        :param regularizer_kwargs:
            Additional keyword arguments passed to the regularizer
        :param dropout:
            The optional dropout probability
        :param unique:
            whether to optimize for calculating representations for same indices only once. This is only useful if the
            calculation of representations is either significantly more expensive than an index-based lookup and
            duplicate indices are expected, e.g., when using negative sampling and large batch sizes
        """
        super().__init__()
        self.max_id = max_id
        self.shape = tuple(upgrade_to_sequence(shape))
        self.normalizer = normalizer_resolver.make_safe(normalizer, normalizer_kwargs)
        self.regularizer = regularizer_resolver.make_safe(regularizer, regularizer_kwargs)
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        if unique is None:
            # heuristic
            unique = not isinstance(self, Embedding)
        self.unique = unique

    @abstractmethod
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get representations for indices, without applying normalization, regularization or output dropout."""
        raise NotImplementedError

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Get representations for indices.

        .. note ::
            depending on :attr:`Representation.unique`, this implementation will use an optimization for duplicate
            indices. It is generally only recommended if computing individual representation is expensive, e.g.,
            since it involves message passing, or a large encoder networks, but discouraged for cheap lookups, e.g., a
            plain embedding lookup.

        :param indices: shape: s
            The indices, or None. If None, this is interpreted as ``torch.arange(self.max_id)`` (although implemented
            more efficiently).

        :return: shape: (``*s``, ``*self.shape``)
            The representations.
        """
        inverse = None
        if indices is not None and self.unique:
            indices, inverse = indices.unique(return_inverse=True)
        x = self._plain_forward(indices=indices)
        # normalize *before* repeating
        if self.normalizer is not None:
            x = self.normalizer(x)
        # repeat if necessary
        if inverse is not None:
            x = x[inverse]
        # regularize *after* repeating
        if self.regularizer is not None:
            self.regularizer.update(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""

    def iter_extra_repr(self) -> Iterable[str]:
        """Iterate over components for :meth:`extra_repr`."""
        yield from super().iter_extra_repr()
        yield f"max_id={self.max_id}"
        yield f"shape={self.shape}"
        yield f"unique={self.unique}"
        if self.normalizer is not None:
            yield f"normalizer={self.normalizer}"
        # dropout & regularizer will appear automatically, since it is a nn.Module

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return get_preferred_device(module=self, allow_ambiguity=True)


class SubsetRepresentation(Representation):
    """A representation module, which only exposes a subset of representations of its base."""

    def __init__(
        self,
        max_id: int,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        shape: Optional[OneOrSequence[int]] = None,
        **kwargs,
    ):
        """
        Initialize the representations.

        :param max_id:
            the maximum number of relations.
        :param base:
            the base representations. have to have a sufficient number of representations, i.e., at least max_id.
        :param base_kwargs:
            additional keyword arguments for the base representation
        :param shape:
            The shape of an individual representation.
        :param kwargs:
            additional keyword-based parameters passed to super.__init__

        :raises ValueError: if ``max_id`` is larger than the base representation's mad_id
        """
        # has to be imported here to avoid cyclic import
        from . import representation_resolver

        base = representation_resolver.make(base, pos_kwargs=base_kwargs)
        if max_id > base.max_id:
            raise ValueError(
                f"Base representations comprise only {base.max_id} representations, "
                f"but at least {max_id} are required.",
            )
        super().__init__(max_id=max_id, shape=ShapeError.verify(shape=base.shape, reference=shape), **kwargs)
        self.base = base

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            indices = torch.arange(self.max_id, device=self.device)
        return self.base._plain_forward(indices=indices)


class Embedding(Representation):
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
    >>> from pykeen.models import ERModel
    >>> model = ERModel(
    ...     triples_factory=dataset.training,
    ...     interaction='distmult',
    ...     entity_representations_kwargs=dict(embedding_dim=3, dropout=0.1),
    ...     relation_representations_kwargs=dict(embedding_dim=3, dropout=0.1),
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
        max_id: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        shape: Union[None, int, Sequence[int]] = None,
        initializer: Hint[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Hint[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Instantiate an embedding with extended functionality.

        .. note ::
            the difference between a *normalizer* (cf. :class:`Representation`) and a *constrainer* is that the
            normalizer is applied to the retrieved representations, and part of the forward call. Thus, it is part
            of the computational graph, and may contribute towards the gradients received by the weight. A
            *constrainer* on the other hand, is applied *after* a parameter update (using the
            :meth:`post_parameter_update` hook), and hence *not* part of the computational graph.

        :param max_id: >0
            The number of embeddings.
        :param num_embeddings: >0
            The number of embeddings.
        :param embedding_dim: >0
            The embedding dimensionality.
        :param shape:
            The shape of an individual representation.
        :param initializer:
            An optional initializer, which takes an uninitialized (num_embeddings, embedding_dim) tensor as input,
            and returns an initialized tensor of same shape and dtype (which may be the same, i.e. the
            initialization may be in-place). Can be passed as a function, or as string corresponding to a key in
            :data:`pykeen.nn.representation.initializers` such as:

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
        :param constrainer:
            A function which is applied to the weights after each parameter update, without tracking gradients.
            It may be used to enforce model constraints outside of gradient-based training. The function does not need
            to be in-place, but the weight tensor is modified in-place. Can be passed as a function, or as a string
            corresponding to a key in :data:`pykeen.nn.representation.constrainers` such as:

            - ``'normalize'``
            - ``'complex_normalize'``
            - ``'clamp'``
            - ``'clamp_norm'``
        :param constrainer_kwargs:
            Additional keyword arguments passed to the constrainer
        :param trainable: Should the wrapped embeddings be marked to require gradient. Defaults to True.
        :param dtype: The datatype (otherwise uses :func:`torch.get_default_dtype` to look up)
        :param kwargs:
            additional keyword-based parameters passed to Representation.__init__
        """
        # normalize num_embeddings vs. max_id
        max_id = process_max_id(max_id, num_embeddings)

        # normalize embedding_dim vs. shape
        _embedding_dim, shape = process_shape(embedding_dim, shape)

        if dtype is None:
            dtype = torch.get_default_dtype()

        # work-around until full complex support (torch==1.10 still does not work)
        # TODO: verify that this is our understanding of complex!
        self.is_complex = dtype.is_complex
        _shape = shape
        if self.is_complex:
            _shape = tuple(shape[:-1]) + (shape[-1], 2)
            _embedding_dim = _embedding_dim * 2
            # note: this seems to work, as finfo returns the datatype of the underlying floating
            # point dtype, rather than the combined complex one
            dtype = getattr(torch, torch.finfo(dtype).dtype)
        self._shape = _shape

        super().__init__(max_id=max_id, shape=shape, **kwargs)

        # use make for initializer since there's a default, and make_safe
        # for the others to pass through None values
        self.initializer = initializer_resolver.make(initializer, initializer_kwargs)
        self.constrainer = constrainer_resolver.make_safe(constrainer, constrainer_kwargs)
        self._embeddings = torch.nn.Embedding(num_embeddings=max_id, embedding_dim=_embedding_dim, dtype=dtype)
        self._embeddings.requires_grad_(trainable)

    # docstr-coverage: inherited
    def reset_parameters(self) -> None:  # noqa: D102
        # initialize weights in-place
        self._embeddings.weight.data = self.initializer(
            self._embeddings.weight.data.view(self.max_id, *self._shape),
        ).view(*self._embeddings.weight.data.shape)

    # docstr-coverage: inherited
    def post_parameter_update(self):  # noqa: D102
        # apply constraints in-place
        if self.constrainer is not None:
            x = self._plain_forward()
            x = self.constrainer(x)
            # fixme: work-around until nn.Embedding supports complex
            if self.is_complex:
                x = torch.view_as_real(x)
            self._embeddings.weight.data = x.view(*self._embeddings.weight.data.shape)

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            prefix_shape = (self.max_id,)
            x = self._embeddings.weight
        else:
            prefix_shape = indices.shape
            x = self._embeddings(indices.to(self.device))
        x = x.view(*prefix_shape, *self._shape)
        # fixme: work-around until nn.Embedding supports complex
        if self.is_complex:
            x = torch.view_as_complex(x)
        # verify that contiguity is preserved
        assert x.is_contiguous()
        return x


class LowRankRepresentation(Representation):
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
        shape: OneOrSequence[int],
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
            additional keyword based arguments passed to :class:`pykeen.nn.representation.Embedding`, which is used
            for the base representations.
        """
        super().__init__(max_id=max_id, shape=shape)
        self.bases = Embedding(max_id=num_bases, shape=shape, **kwargs)
        self.weight_initializer = weight_initializer
        self.weight = nn.Parameter(torch.empty(max_id, num_bases))
        self.reset_parameters()

    @classmethod
    def approximate(cls, other: Representation, **kwargs) -> "LowRankRepresentation":
        """
        Construct a low-rank approximation of another representation.

        .. note ::

            While this method tries to find a good approximation of the base representation, you may lose all (useful)
            inductive biases you had with the original one, e.g., from shared tokens in
            :class:`pykeen.representation.NodePieceRepresentation`.

        :param other:
            the other representation
        :param kwargs:
            additional keyword-based parameters passed to :meth:`LowRankRepresentation.__init__`. Must not contain
            `max_id` nor `shape`, which are determined by `other`

        :return:
            a low-rank approximation obtained via (truncated) SVD
        """
        # create low-rank approximation object
        r = cls(max_id=other.max_id, shape=other.shape, **kwargs)
        # get base representations, shape: (n, *ds)
        x = other(indices=None)
        # calculate SVD, U.shape: (n, k), s.shape: (k,), u.shape: (k, prod(ds))
        u, s, vh = torch.svd_lowrank(x.view(x.shape[0], -1), q=r.num_bases)
        # overwrite bases and weights
        r.bases._embeddings.weight.data = vh
        r.weight.data = torch.einsum("nk, k -> nk", u, s)
        return r

    # docstr-coverage: inherited
    def reset_parameters(self) -> None:  # noqa: D102
        self.bases.reset_parameters()
        self.weight.data = self.weight_initializer(self.weight)

    @property
    def num_bases(self) -> int:
        """Return the number of bases."""
        return self.bases.max_id

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # get all base representations, shape: (num_bases, *shape)
        bases = self.bases(indices=None)
        # get base weights, shape: (*batch_dims, num_bases)
        weight = self.weight
        if indices is not None:
            weight = weight[indices.to(self.device)]
        # weighted linear combination of bases, shape: (*batch_dims, *shape)
        return torch.tensordot(weight, bases, dims=([-1], [0]))


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


def process_max_id(max_id: Optional[int], num_embeddings: Optional[int]) -> int:
    """Normalize max_id."""
    if max_id is None:
        if num_embeddings is None:
            raise ValueError("Must provide max_id")
        warnings.warn("prefer using 'max_id' over 'num_embeddings'", DeprecationWarning)
        max_id = num_embeddings
    elif num_embeddings is not None and num_embeddings != max_id:
        raise ValueError("Cannot provide both, 'max_id' over 'num_embeddings'")
    return max_id


constrainer_resolver = FunctionResolver([functional.normalize, complex_normalize, torch.clamp, clamp_norm])

normalizer_resolver = FunctionResolver([functional.normalize])


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
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        activation: HintOrType[nn.Module] = nn.Identity,
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
        :param attention_heads:
            Number of attention heads when using the attention weighting
        :param attention_dropout:
            Dropout for the attention message weighting
        :param activation:
            The activation to use.
        :param activation_kwargs:
            Additional key-word based arguments passed to the activation.
        :param edge_weighting:
            A pre-instantiated :class:`EdgeWeighting`, a class, or name to look
            up with :class:`class_resolver`.
        """
        super().__init__()

        # normalize output dimension
        output_dim = output_dim or input_dim

        # entity-relation composition
        self.composition = composition_resolver.make(composition)

        # edge weighting
        self.edge_weighting: EdgeWeighting = edge_weight_resolver.make(
            edge_weighting, message_dim=output_dim, dropout=attention_dropout, num_heads=attention_heads
        )

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
        m = self.edge_weighting(source=source, target=target, message=m, x_e=x_e)

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


def build_representation(
    max_id: int,
    representation: HintOrType[Representation],
    representation_kwargs: OptionalKwargs,
) -> Representation:
    """Build representations and check maximum ID."""
    # has to be imported here to avoid cyclic imports
    from . import representation_resolver

    representation = representation_resolver.make(
        representation,
        pos_kwargs=representation_kwargs,
        # kwargs
        max_id=max_id,
    )
    if representation.max_id != max_id:
        raise ValueError(
            f"Representations should provide {max_id} representations, " f"but have {representation.max_id}",
        )
    return representation


class CombinedCompGCNRepresentations(nn.Module):
    """A sequence of CompGCN layers."""

    # Buffered enriched entity and relation representations
    enriched_representations: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        entity_representations: HintOrType[Representation] = None,
        entity_representations_kwargs: OptionalKwargs = None,
        relation_representations: HintOrType[Representation] = None,
        relation_representations_kwargs: OptionalKwargs = None,
        num_layers: Optional[int] = 1,
        dims: Union[None, int, Sequence[int]] = None,
        layer_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the combined entity and relation representation module.

        :param triples_factory:
            The triples factory containing the training triples.
        :param entity_representations:
            the base entity representations
        :param entity_representations_kwargs:
            additional keyword parameters for the base entity representations
        :param relation_representations:
            the base relation representations
        :param relation_representations_kwargs:
            additional keyword parameters for the base relation representations
        :param num_layers:
            The number of message passing layers to use. If None, will be inferred by len(dims), i.e., requires dims to
            be a sequence / list.
        :param dims:
            The hidden dimensions to use. If None, defaults to the embedding dimension of the base representations.
            If an integer, is the same for all layers. The last dimension is equal to the output dimension.
        :param layer_kwargs:
            Additional key-word based parameters passed to the individual layers; cf. CompGCNLayer.
        :raises ValueError: for several invalid combinations of arguments:
            1. If the dimensions were given as an integer but no number of layers were given
            2. If the dimensions were given as a ist but it does not match the number of layers that were given
        """
        super().__init__()
        # TODO: Check
        assert triples_factory.create_inverse_triples
        self.entity_representations = build_representation(
            max_id=triples_factory.num_entities,
            representation=entity_representations,
            representation_kwargs=entity_representations_kwargs,
        )
        self.relation_representations = build_representation(
            max_id=2 * triples_factory.real_num_relations,
            representation=relation_representations,
            representation_kwargs=relation_representations_kwargs,
        )
        if len(self.entity_representations.shape) > 1:
            raise ValueError(f"{self.__class__.__name__} requires vector base entity representations.")
        input_dim = self.entity_representations.shape[0]
        # TODO: might not be true for all compositions
        if self.relation_representations.shape != self.entity_representations.shape:
            raise ValueError(
                f"{self.__class__.__name__} requires entity and relation representations of the same shape."
            )

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
        for input_dim_, output_dim in zip(itertools.chain([input_dim], dims), dims):
            layers.append(
                CompGCNLayer(
                    input_dim=input_dim_,
                    output_dim=output_dim,
                    **(layer_kwargs or {}),
                )
            )
        self.layers = nn.ModuleList(layers)

        # register buffers for adjacency matrix; we use the same format as PyTorch Geometric
        # TODO: This always uses all training triples for message passing
        self.register_buffer(name="edge_index", tensor=get_edge_index(triples_factory=triples_factory))
        self.register_buffer(name="edge_type", tensor=triples_factory.mapped_triples[:, 1])

        # initialize buffer of enriched representations
        self.enriched_representations = None

    # docstr-coverage: inherited
    def post_parameter_update(self) -> None:  # noqa: D102
        # invalidate enriched embeddings
        self.enriched_representations = None

    # docstr-coverage: inherited
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


class SingleCompGCNRepresentation(Representation):
    """A wrapper around the combined representation module."""

    def __init__(
        self,
        combined: CombinedCompGCNRepresentations,
        position: int = 0,
        shape: Optional[OneOrSequence[int]] = None,
        **kwargs,
    ):
        """
        Initialize the module.

        :param combined:
            The combined representations.
        :param position:
            The position, either 0 for entities, or 1 for relations.
        :param shape:
            The shape of an individual representation.
        :param kwargs:
            additional keyword-based parameters passed to super.__init__
        :raises ValueError: If an invalid value is given for the position
        """
        if position == 0:  # entity
            max_id = combined.entity_representations.max_id
            shape_ = (combined.output_dim,)
        elif position == 1:  # relation
            max_id = combined.relation_representations.max_id
            shape_ = (combined.output_dim,)
        else:
            raise ValueError
        super().__init__(max_id=max_id, shape=ShapeError.verify(shape=shape_, reference=shape), **kwargs)
        self.combined = combined
        self.position = position
        self.reset_parameters()

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        x = self.combined()[self.position]
        if indices is not None:
            x = x[indices.to(self.device)]
        return x


def _clean_labels(labels: Sequence[Optional[str]], missing_action: Literal["error", "blank"]) -> Sequence[str]:
    if missing_action == "error":
        idx = [i for i, label in enumerate(labels) if label is None]
        if idx:
            raise ValueError(
                f"The labels at the following indexes were none. "
                f"Consider an alternate `missing_action` policy.\n{idx}",
            )
        return cast(Sequence[str], labels)
    elif missing_action == "blank":
        return [label or "" for label in labels]
    else:
        raise ValueError(f"Invalid `missing_action` policy: {missing_action}")


class TextRepresentation(Representation):
    """
    Textual representations using a text encoder on labels.

    Example Usage:

    Entity representations are obtained by encoding the labels with a Transformer model. The transformer
    model becomes part of the KGE model, and its parameters are trained jointly.

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.nn.representation import TextRepresentation
        from pykeen.models import ERModel

        dataset = get_dataset(dataset="nations")
        entity_representations = TextRepresentation.from_dataset(
            dataset=dataset,
            encoder="transformer",
        )
        model = ERModel(
            interaction="ermlp",
            entity_representations=entity_representations,
            relation_representations_kwargs=dict(shape=entity_representations.shape),
        )
    """

    labels: List[str]

    def __init__(
        self,
        labels: Sequence[Optional[str]],
        max_id: Optional[int] = None,
        shape: Optional[OneOrSequence[int]] = None,
        encoder: HintOrType[TextEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        missing_action: Literal["blank", "error"] = "error",
        **kwargs,
    ):
        """
        Initialize the representation.

        :param labels:
            an ordered, finite collection of labels
        :param max_id:
            the number of representations. If provided, has to match the number of labels
        :param shape:
            The shape of an individual representation.
        :param encoder:
            the text encoder, or a hint thereof
        :param encoder_kwargs:
            keyword-based parameters used to instantiate the text encoder
        :param missing_action:
            Which policy for handling nones in the given labels. If "error", raises an error
            on any nones. If "blank", replaces nones with an empty string.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`

        :raises ValueError:
            if the max_id does not match
        """
        encoder = text_encoder_resolver.make(encoder, encoder_kwargs)
        # check max_id
        max_id = max_id or len(labels)
        if max_id != len(labels):
            raise ValueError(f"max_id={max_id} does not match len(labels)={len(labels)}")
        labels = _clean_labels(labels, missing_action)
        # infer shape
        shape = ShapeError.verify(shape=encoder.encode_all(labels[0:1]).shape[1:], reference=shape)
        super().__init__(max_id=max_id, shape=shape, **kwargs)
        self.labels = list(labels)
        # assign after super, since they should be properly registered as submodules
        self.encoder = encoder

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "TextRepresentation":
        """
        Prepare a text representations with labels from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :meth:`TextRepresentation.__init__`

        :returns:
            a text representation from the triples factory
        """
        labeling: Labeling = triples_factory.entity_labeling if for_entities else triples_factory.relation_labeling
        return cls(labels=labeling.all_labels(), **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        **kwargs,
    ) -> "TextRepresentation":
        """Prepare text representation with labels from a dataset.

        :param dataset:
            the dataset
        :param kwargs:
            additional keyword-based parameters passed to
            :meth:`TextRepresentation.from_triples_factory`

        :return:
            a text representation from the dataset

        :raises TypeError:
            if the dataset's triples factory does not provide labels
        """
        if not isinstance(dataset.training, TriplesFactory):
            raise TypeError(f"{cls.__name__} requires access to labels, but dataset.training does not provide such.")
        return cls.from_triples_factory(triples_factory=dataset.training, **kwargs)

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            labels = self.labels
        else:
            labels = [self.labels[i] for i in indices.tolist()]
        return self.encoder(labels=labels)


class CombinedRepresentation(Representation):
    """A combined representation."""

    #: the base representations
    base: Sequence[Representation]

    #: the combination module
    combination: Combination

    def __init__(
        self,
        max_id: int,
        shape: Optional[OneOrSequence[int]] = None,
        base: OneOrManyHintOrType[Representation] = None,
        base_kwargs: OneOrManyOptionalKwargs = None,
        combination: HintOrType[Combination] = None,
        combination_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param max_id:
            the number of representations.
        :param shape:
            The shape of an individual representation.
        :param base:
            the base representations, or hints thereof
        :param base_kwargs:
            keyword-based parameters for the instantiation of base representations
        :param combination:
            the combination, or a hint thereof
        :param combination_kwargs:
            additional keyword-based parameters used to instantiate the combination
        :param kwargs:
            additional keyword-based parameters passed to `Representation.__init__`.
            May not contain any of `{max_id, shape, unique}`.

        :raises ValueError:
            if the `max_id` of the base representations does not match
        """
        # input normalization
        combination = combination_resolver.make(combination, combination_kwargs)

        # has to be imported here to avoid cyclic import
        from . import representation_resolver

        # create base representations
        base = representation_resolver.make_many(base, kwargs=base_kwargs, max_id=max_id)

        # verify same ID range
        max_ids = sorted(set(b.max_id for b in base))
        if len(max_ids) != 1:
            # note: we could also relax the requiremen, and set max_id = min(max_ids)
            raise ValueError(f"Maximum number of Ids does not match! {max_ids}")
        max_id = max_id or max_ids[0]
        if max_id != max_ids[0]:
            raise ValueError(f"max_id={max_id} does not match base max_id={max_ids[0]}")

        # shape inference
        shape = ShapeError.verify(shape=combination.output_shape(input_shapes=[b.shape for b in base]), reference=shape)
        super().__init__(max_id=max_id, shape=shape, unique=any(b.unique for b in base), **kwargs)

        # assign base representations *after* super init
        self.base = nn.ModuleList(base)
        self.combination = combination

    @staticmethod
    def combine(
        combination: nn.Module, base: Sequence[Representation], indices: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Combine base representations for the given indices.

        :param combination: the combination
        :param base: the base representations
        :param indices: the indices, as given to :meth:`Representation._plain_forward`

        :return:
            the combined representations for the given indices
        """
        return combination([b._plain_forward(indices=indices) for b in base])

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.combine(combination=self.combination, base=self.base, indices=indices)


class CachedTextRepresentation(TextRepresentation):
    """Textual representations for datasets with identifiers that can be looked up with a :class:`TextCache`."""

    cache_cls: ClassVar[Type[TextCache]]

    def __init__(self, identifiers: Sequence[str], cache: TextCache | None = None, **kwargs):
        """
        Initialize the representation.

        :param identifiers:
            the IDs to be resolved by the class, e.g., wikidata IDs. for :class:`WikidataTextRepresentation`,
            biomedical entities represented as compact URIs (CURIEs) for :class:`BiomedicalCURIERepresentation`
        :param cache:
            a pre-instantiated text cache. If None, :attr:`cache_cls` is used to instantiate one.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`TextRepresentation.__init__`
        """
        cache = self.cache_cls() if cache is None else cache
        labels = cache.get_texts(identifiers=identifiers)
        # delegate to super class
        super().__init__(labels=labels, **kwargs)

    # docstr-coverage: inherited
    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "TextRepresentation":  # noqa: D102
        labeling: Labeling = triples_factory.entity_labeling if for_entities else triples_factory.relation_labeling
        return cls(identifiers=labeling.all_labels(), **kwargs)


class WikidataTextRepresentation(CachedTextRepresentation):
    """
    Textual representations for datasets grounded in Wikidata.

    The label and description for each entity are obtained from Wikidata using
    :class:`pykeen.nn.utils.WikidataCache` and encoded with :class:`TextRepresentation`.

    Example usage:

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.models import ERModel
        from pykeen.nn import WikidataTextRepresentation
        from pykeen.pipeline import pipeline

        dataset = get_dataset(dataset="codexsmall")
        entity_representations = WikidataTextRepresentation.from_dataset(dataset=dataset, encoder="transformer")
        result = pipeline(
            dataset=dataset,
            model=ERModel,
            model_kwargs=dict(
                interaction="distmult",
                entity_representations=entity_representations,
                relation_representation_kwargs=dict(
                    shape=entity_representations.shape,
                ),
            ),
        )
    """

    cache_cls = WikidataCache


class BiomedicalCURIERepresentation(CachedTextRepresentation):
    """
    Textual representations for datasets grounded with biomedical CURIEs.

    The label and description for each entity are obtained via :mod:`pyobo` using
    :class:`pykeen.nn.utils.PyOBOCache` and encoded with :class:`TextRepresentation`.

    Example usage:

    .. code-block:: python

        from pykeen.datasets import get_dataset
        from pykeen.models import ERModel
        from pykeen.nn import BiomedicalCURIERepresentation
        from pykeen.pipeline import pipeline
        import bioontologies

        # Generate graph dataset from the Monarch Disease Ontology (MONDO)
        graph = bioontologies.get_obograph_by_prefix("mondo").squeeze(standardize=True)
        triples = (edge.as_tuple() for edge in graph.edges)
        triples = [t for t in triples if all(t)]
        triples = TriplesFactory.from_labeled_triples(np.array(triples))
        dataset = Dataset.from_tf(triples)

        entity_representations = BiomedicalCURIERepresentation.from_dataset(
            dataset=dataset, encoder="transformer",
        )
        result = pipeline(
            dataset=dataset,
            model=ERModel,
            model_kwargs=dict(
                interaction="distmult",
                entity_representations=entity_representations,
                relation_representation_kwargs=dict(
                    shape=entity_representations.shape,
                ),
            ),
        )
    """

    cache_cls = PyOBOCache


class PartitionRepresentation(Representation):
    """
    A partition of the indices into different representation modules.

    Each index is assigned to an index in exactly one of the base representations. This representation is useful, e.g.,
    when one of the base representations cannot provide vectors for each of the indices, and another representation is
    used as back-up.

    Consider the following example: We only have textual information for two entities. We want to use textual features
    computed from them, which should not be trained. For the remaining entities we want to use directly trainable
    embeddings.

    We start by creating the representation for those entities where we have labels:

    >>> from pykeen.nn import Embedding, init
    >>> num_entities = 5
    >>> labels = {1: "a first description", 4: "a second description"}
    >>> label_initializer = init.LabelBasedInitializer(labels=list(labels.values()))
    >>> label_repr = label_initializer.as_embedding()

    Next, we create representations for the remaining ones

    >>> non_label_repr = Embedding(max_id=num_entities - len(labels), shape=label_repr.shape)

    To combine them into a single representation module we first need to define the assignment, i.e., where to look-up
    the global ids. For this, we create a tensor of shape `(num_entities, 2)`, with the index of the base
    representation, and the *local* index inside this representation

    >>> import torch
    >>> assignment = torch.as_tensor([(1, 0), (0, 0), (1, 1), (1, 2), (0, 1)])
    >>> from pykeen.nn import PartitionRepresentation
    >>> entity_repr = PartitionRepresentation(assignment=assignment, bases=[label_repr, non_label_repr])

    For brevity, we use here randomly generated triples factories instead of the actual data

    >>> from pykeen.triples.generation import generate_triples_factory
    >>> training = generate_triples_factory(num_entities=num_entities, num_relations=5, num_triples=31)
    >>> testing = generate_triples_factory(num_entities=num_entities, num_relations=5, num_triples=17)

    The combined representation can now be used as any other representation, e.g., to train a DistMult model:

    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models import ERModel
    >>> pipeline(
    ...     model=ERModel,
    ...     interaction="distmult",
    ...     model_kwargs=dict(
    ...         entity_representation=entity_repr,
    ...         relation_representation_kwargs=dict(shape=shape),
    ...     ),
    ...     training=training,
    ...     testing=testing,
    ... )
    """

    #: the assignment from global ID to (representation, local id), shape: (max_id, 2)
    assignment: torch.LongTensor

    def __init__(
        self,
        assignment: torch.LongTensor,
        shape: Optional[OneOrSequence[int]] = None,
        bases: OneOrSequence[HintOrType[Representation]] = None,
        bases_kwargs: OneOrSequence[OptionalKwargs] = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        .. warning ::
            the base representations have to have coherent shapes

        :param assignment: shape: (max_id, 2)
            the assignment, as tuples `(base_id, local_id)`, where `base_id` refers to the index of the base
            representation and `local_id` is an index used to lookup in the base representation
        :param shape:
            the shape of an individual representation. If provided, must match the bases' shape
        :param bases:
            the base representations, or hints thereof.
        :param bases_kwargs:
            keyword-based parameters to instantiate the base representations
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`. May not contain `max_id`,
            or `shape`, which are inferred from the base representations.

        :raises ValueError:
            if any of the inputs is invalid
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        # instantiate base representations if necessary
        bases = representation_resolver.make_many(bases, bases_kwargs)

        # there needs to be at least one base
        if not bases:
            raise ValueError("Must provide at least one base representation")
        # while possible, this might be unintended
        if len(bases) == 1:
            logger.warning(f"Encountered only a single base representation: {bases[0]}")

        # extract shape
        shapes = [base.shape for base in bases]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent base shapes: {shapes}")
        shape = ShapeError.verify(shape=shapes[0], reference=shape)

        # check for invalid base ids
        unknown_base_ids = set(assignment[:, 0].tolist()).difference(range(len(bases)))
        if unknown_base_ids:
            raise ValueError(f"Invalid representation Ids in assignment: {unknown_base_ids}")

        # check for invalid local indices
        for i, base in enumerate(bases):
            max_index = assignment[assignment[:, 0] == i, 1].max().item()
            if max_index >= base.max_id:
                raise ValueError(f"base {base} (index:{i}) cannot provide indices up to {max_index}")

        super().__init__(max_id=assignment.shape[0], shape=shape, **kwargs)

        # assign modules / buffers *after* super init
        self.bases = bases
        self.register_buffer(name="assignment", tensor=assignment)

    # docstr-coverage: inherited
    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        assignment = self.assignment
        if indices is not None:
            assignment = assignment[indices]
        # flatten assignment to ease construction of inverse indices
        prefix_shape = assignment.shape[:-1]
        assignment = assignment.view(-1, 2)
        # we group indices by the representation which provides them
        # thus, we need an inverse to restore the correct order
        inverse = torch.empty_like(assignment[:, 0])
        xs = []
        offset = 0
        for i, base in enumerate(self.bases):
            mask = assignment[:, 0] == i
            # get representations
            local_indices = assignment[:, 1][mask]
            xs.append(base(indices=local_indices))
            # update inverse indices
            end = offset + local_indices.numel()
            inverse[mask] = torch.arange(offset, end, device=inverse.device)
            offset = end
        x = torch.cat(xs, dim=0)[inverse]
        # invert flattening
        if len(prefix_shape) != 1:
            x = x.view(*prefix_shape, *x.shape[1:])
        return x


class BackfillRepresentation(PartitionRepresentation):
    """A variant of a partition representation that is easily applicable to a single base representation.

    Similarly to the :mod:`PartitionRepresentation` representation example, we start by
    creating the representation for those entities where we have labels:

    >>> from pykeen.nn import Embedding, init
    >>> num_entities = 5
    >>> labels = {1: "a first description", 4: "a second description"}
    >>> label_initializer = init.LabelBasedInitializer(labels=list(labels.values()))
    >>> shape = label_initializer.tensor.shape[1:]
    >>> label_repr = Embedding(max_id=len(labels), shape=shape, initializer=label_initializer, trainable=False)

    Next, we directly create representations for the remaining ones using the backfill representation.
    To do this, we need to create an iterable (e.g., a set) of all of the entity IDs that are in the base
    representation. Then, the assignments to the base representation and an auxillary representation are
    automatically generated for the base class

    >>> from pykeen.nn import BackfillRepresentation
    >>> entity_repr = BackfillRepresentation(base_ids=set(labels), max_id=num_entities, base=label_repr)

    For brevity, we use here randomly generated triples factories instead of the actual data
    >>> from pykeen.triples.generation import generate_triples_factory
    >>> training = generate_triples_factory(num_entities=num_entities, num_relations=5, num_triples=31)
    >>> testing = generate_triples_factory(num_entities=num_entities, num_relations=5, num_triples=17)
    The combined representation can now be used as any other representation, e.g., to train a DistMult model:
    >>> from pykeen.pipeline import pipeline
    >>> from pykeen.models import ERModel
    >>> pipeline(
    ...     model=ERModel,
    ...     interaction="distmult",
    ...     model_kwargs=dict(
    ...         entity_representation=entity_repr,
    ...         relation_representation_kwargs=dict(shape=shape),
    ...     ),
    ...     training=training,
    ...     testing=testing,
    ... )
    """

    def __init__(
        self,
        max_id: int,
        base_ids: Iterable[int],
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        backfill: HintOrType[Representation] = None,
        backfill_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """Initialize the representation.

        :param max_id:
            The total number of entities that need to be embedded
        :param base_ids:
            An iterable of integer entity indexes which are provided through the base representations
        :param base:
            the base representation, or a hint thereof.
        :param base_kwargs:
            keyword-based parameters to instantiate the base representation
        :param backfill:
            the backfill representation, or hints thereof.
        :param backfill_kwargs:
            keyword-based parameters to instantiate the backfill representation
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`. May not contain `max_id`,
            or `shape`, which are inferred from the base representations.
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        base_ids = sorted(set(base_ids))
        base = representation_resolver.make(base, base_kwargs, max_id=len(base_ids))
        # comment: not all representations support passing a shape parameter
        backfill = representation_resolver.make(
            backfill, backfill_kwargs, max_id=max_id - base.max_id, shape=base.shape
        )

        # create assignment
        assignment = torch.full(size=(max_id, 2), fill_value=1, dtype=torch.long)
        # base
        assignment[base_ids, 0] = 0
        assignment[base_ids, 1] = torch.arange(base.max_id)
        # other
        mask = torch.ones(assignment.shape[0], dtype=torch.bool)
        mask[base_ids] = False
        assignment[mask, 0] = 1
        assignment[mask, 1] = torch.arange(backfill.max_id)

        super().__init__(assignment=assignment, bases=[base, backfill], **kwargs)


class TransformedRepresentation(Representation):
    """
    A (learnable) transformation upon base representations.

    In the following example, we create representations which are obtained from a trainable transformation of fixed
    random walk encoding features. We first load the dataset, here Nations:

    >>> from pykeen.datasets import get_dataset
    >>> dataset = get_dataset(dataset="nations")

    Next, we create a random-walk positional encoding of dimension 32:

    >>> from pykeen.nn import init
    >>> dim = 32
    >>> initializer = init.RandomWalkPositionalEncoding(triples_factory=dataset.training, dim=dim+1)
    We used dim+1 for the RWPE initializion as by default it doesn't return the first dimension of 0's
    That is, in the default setup, dim = 33 would return a 32d vector

    For the transformation, we use a simple 2-layer MLP

    >>> from torch import nn
    >>> hidden = 64
    >>> mlp = nn.Sequential(
    ...     nn.Linear(in_features=dim, out_features=hidden),
    ...     nn.ReLU(),
    ...     nn.Linear(in_features=hidden, out_features=dim),
    ... )

    Finally, the transformed representation is given as

    >>> from pykeen.nn import TransformedRepresentation
    >>> r = TransformedRepresentation(
    ...     transformation=mlp,
    ...     base_kwargs=dict(max_id=dataset.num_entities, shape=(dim,), initializer=initializer, trainable=False),
    ... )
    """

    def __init__(
        self,
        transformation: nn.Module,
        max_id: Optional[int] = None,
        shape: Optional[OneOrSequence[int]] = None,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param transformation:
            the transformation
        :param max_id:
            the number of representations. If provided, must match the base max id
        :param shape:
            the individual representations' shape. If provided, must match the output shape of the transformation
        :param base:
            the base representation, or a hint thereof, cf. `representation_resolver`
        :param base_kwargs:
            keyword-based parameters used to instantiate the base representation
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`.

        :raises ValueError:
            if the max_id or shape does not match
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        base = representation_resolver.make(base, base_kwargs)

        # infer shape
        shape = ShapeError.verify(
            shape=self._help_forward(
                base=base, transformation=transformation, indices=torch.zeros(1, dtype=torch.long, device=base.device)
            ).shape[1:],
            reference=shape,
        )
        # infer max_id
        max_id = max_id or base.max_id
        if max_id != base.max_id:
            raise ValueError(f"Incompatible max_id={max_id} vs. base.max_id={base.max_id}")

        super().__init__(max_id=max_id, shape=shape, **kwargs)
        self.transformation = transformation
        self.base = base

    @staticmethod
    def _help_forward(
        base: Representation, transformation: nn.Module, indices: Optional[torch.LongTensor]
    ) -> torch.FloatTensor:
        """
        Obtain base representations and apply the transformation.

        :param base:
            the base representation module
        :param transformation:
            the transformation
        :param indices:
            the indices

        :return:
            the transformed base representations
        """
        return transformation(base(indices=indices))

    # docstr-coverage: inherited
    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        return self._help_forward(base=self.base, transformation=self.transformation, indices=indices)


# TODO: can be a combined representations, with appropriate tensor-train combination
class TensorTrainRepresentation(Representation):
    r"""
    A tensor factorization of representations.

    In the simple case without provided assignment this corresponds to `TT-emb` described in
    https://assets.amazon.science/5c/0f/dd3eb08c4df88f2b4722e5fa8a7c/nimble-gnn-embedding-with-tensor-train-decomposition.pdf

    where

    .. math ::

        \mathbf{A}[i_1 \cdot \ldots \cdot i_k, j_1 \cdot \ldots \cdot j_k]
            = \sum_{r_i, \ldots, r_k} \mathbf{G}_1[0, i_1, j_1, r_1]
                \cdot \mathbf{G}_2[r_1, i_2, j_2, r_2]
                \cdot \ldots
                \cdot \mathbf{G}_k[r_k, i_k, j_k, 0]

    with TT core $\mathbf{G}_i$ of shape $R_{i-1} \times m_i \times n_i \times R_i$ and $R_0 = R_d = 1$.

    Another variant in the paper used an assignment based on hierarchical topological clustering.
    """

    #: shape: (max_id, num_cores)
    assignment: torch.LongTensor

    #: the bases, length: num_cores, with compatible shapes
    bases: Sequence[Representation]

    @classmethod
    def factor_sizes(cls, max_id: int, shape: Sequence[int], num_cores: int) -> Tuple[Sequence[int], Sequence[int]]:
        r"""Factor the representation shape into smaller shapes for the cores.

        :param max_id:
            the number of representations, "row count", $M$
        :param shape:
            the shape of an individual representation, "column count", $N$
        :param num_cores:
            the number of cores, $k$

        :return:
            a tuple (ms, ns) of positive integer sequences of length $k$ fulfilling

            .. math ::

                \prod \limits_{m_i \in ms} m_i \geq M

                \prod \limits_{n_i \in ns} n_i \geq N
        """
        m_k = int(math.ceil(max_id ** (1 / num_cores)))
        n_k = int(math.ceil(numpy.prod(shape) ** (1 / num_cores)))
        return [m_k] * num_cores, [n_k] * num_cores

    @staticmethod
    def check_assignment(assignment: torch.Tensor, max_id: int, num_cores: int, ms: Sequence[int]):
        """
        Check that the assignment matches the other properties.

        :param assignment: shape: (max_id, num_cores)
            the assignment
        :param max_id:
            the number of representations
        :param num_cores:
            the number of tensor-train cores
        :param ms:
            the individual sizes $m_i$

        :raises ValueError:
            if the assignment is invalid
        """
        # check shape
        if assignment.shape != (max_id, num_cores):
            raise ValueError(
                f"Invalid assignment. Expected shape (max_id, num_cores)={(max_id, num_cores)}, "
                f"but got assignment.shape={assignment.shape}",
            )
        # check value range
        low, high = assignment.min(dim=0).values, assignment.max(dim=0).values
        if (low < 0).any() or (high >= torch.as_tensor(ms, dtype=torch.long)).any():
            raise ValueError(
                f"Invalid values inside assignment: ms={ms} vs. assignment.min(dim=0)={low} "
                f"and assignment.max(dim=0)={high}",
            )

    @staticmethod
    def get_shapes_and_einsum_eq(ranks: Sequence[int], ns: Sequence[int]) -> Tuple[str, Sequence[Tuple[int, ...]]]:
        """
        Determine core shapes and einsum equation.

        :param ranks:
            the core ranks
        :param ns:
            the sizes $n_i$
        :return:
            a pair (eq, shapes), where `eq` is a valid einsum equation and `shapes` a sequence of representation
            shapes. Notice that the shapes do not include the "`max_id` dimension" of the resulting embedding.
        """
        shapes: List[List[int]] = []
        terms: List[List[str]] = []
        out_term: List[str] = ["..."]
        i = 0
        for n_i, (rank_in, rank_out) in zip(ns, more_itertools.pairwise([None, *ranks, None])):
            shape = []
            term = ["..."]

            if rank_in is not None:
                shape.append(rank_in)
                term.append(string.ascii_lowercase[i])
                i += 1

            shape.append(n_i)
            term.append(string.ascii_lowercase[i])
            out_term.append(string.ascii_lowercase[i])
            i += 1

            if rank_out is not None:
                shape.append(rank_out)
                term.append(string.ascii_lowercase[i])
                # do not increase counter i, since the dimension is shared with the following term
                # i += 1

            terms.append(term)
            shapes.append(shape)
        eq = " ".join((", ".join("".join(term) for term in terms), "->", "".join(out_term)))
        return eq, [tuple(shape) for shape in shapes]

    @staticmethod
    def create_default_assignment(max_id: int, num_cores: int, ms: Sequence[int]) -> torch.LongTensor:
        """
        Create an assignment without using structural information.

        :param max_id:
            the number of representations
        :param num_cores:
            the number of tensor cores
        :param ms:
            the sizes $m_i$

        :return: shape: (max_id, num_cores)
            the assignment
        """
        assignment = torch.empty(max_id, num_cores, dtype=torch.long)
        ids = torch.arange(max_id)
        for i, m_i in enumerate(ms):
            assignment[:, i] = ids % m_i
            # ids //= m_i
            ids = torch.div(ids, m_i, rounding_mode="floor")
        return assignment

    @staticmethod
    def check_factors(ms: Sequence[int], ns: Sequence[int], max_id: int, shape: Tuple[int, ...], num_cores: int):
        r"""
        Check whether the factors match the other parts.

        Verifies that

        .. math ::
            \prod \limits_{m_i \in ms} m_i \geq M
            \prod \limits_{n_i \in ns} n_i \geq N

        :param ms: length: num_cores
            the $M$ factors $m_i$
        :param ns: length: num_cores
            the $N$ factors $n_i$
        :param max_id:
            the maximum id, $M$
        :param shape:
            the shape, $N=prod(shape)$
        :param num_cores:
            the number of cores

        :raises ValueError:
            if any of the conditions is violated
        """
        if len(ms) != num_cores or len(ns) != num_cores:
            raise ValueError(f"Invalid length: len(ms)={len(ms)}, len(ns)={len(ns)} vs. num_cores={num_cores}")

        m_prod = numpy.prod(ms).item()
        if m_prod < max_id:
            raise ValueError(f"prod(ms)={m_prod} < max_id={max_id}")

        n_prod = numpy.prod(ns).item()
        s_prod = numpy.prod(shape).item()
        if n_prod < s_prod:
            raise ValueError(f"prod(ns)={n_prod} < prod(shape)={s_prod}")

    def __init__(
        self,
        assignment: Optional[torch.LongTensor] = None,
        num_cores: int = 3,
        ranks: OneOrSequence[int] = 2,
        bases: OneOrManyHintOrType = None,
        bases_kwargs: OneOrManyOptionalKwargs = None,
        **kwargs,
    ) -> None:
        """Initialize the representation.

        :param assignment: shape: (max_id, num_cores)
            the assignment on each level
        :param num_cores:
            the number of cores to use
        :param ranks: length: num_cores - 1
            the individual ranks. Note that $R_0 = R_d = 1$ should not be included
        :param bases:
            the base representations for each level, or hints thereof.
        :param bases_kwargs:
            keyword-based parameters for the bases
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`

        :raises ValueError:
            if the input validation on ranks or assignment failed
        """
        # import here to avoid cyclic import
        from . import representation_resolver

        super().__init__(**kwargs)

        # normalize ranks
        ranks = list(upgrade_to_sequence(ranks))
        if len(ranks) == 1:
            ranks = ranks * (num_cores - 1)
        if len(ranks) != num_cores - 1:
            raise ValueError(f"Inconsistent number of ranks {len(ranks)} for num_cores={num_cores}")

        # determine M_k, N_k
        # TODO: allow to pass them from outside?
        ms, ns = self.factor_sizes(max_id=self.max_id, shape=self.shape, num_cores=num_cores)
        self.check_factors(ms, ns, max_id=self.max_id, shape=self.shape, num_cores=num_cores)

        # normalize assignment
        if assignment is None:
            assignment = self.create_default_assignment(max_id=self.max_id, num_cores=num_cores, ms=ms)
        self.check_assignment(assignment=assignment, max_id=self.max_id, num_cores=num_cores, ms=ms)
        self.register_buffer(name="assignment", tensor=assignment)

        # determine shapes and einsum equation
        self.eq, shapes = self.get_shapes_and_einsum_eq(ranks=ranks, ns=ns)

        # create base representations
        self.bases = nn.ModuleList(
            representation_resolver.make(base, base_kwargs, max_id=m_i, shape=shape)
            for base, base_kwargs, m_i, shape in zip(*broadcast_upgrade_to_sequences(bases, bases_kwargs, ms, shapes))
        )

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"num_cores={len(self.bases)}"
        yield f"eq='{self.eq}'"

    # docstr-coverage: inherited
    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        assignment = self.assignment
        if indices is not None:
            assignment = assignment[indices]
        return einsum(self.eq, *(base(indices) for indices, base in zip(assignment.unbind(dim=-1), self.bases))).view(
            *assignment.shape[:-1], *self.shape
        )
