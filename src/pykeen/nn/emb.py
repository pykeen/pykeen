# -*- coding: utf-8 -*-

"""Embedding modules."""

from __future__ import annotations

import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn
from class_resolver import FunctionResolver
from class_resolver.contrib.torch import activation_resolver
from torch import nn
from torch.nn import functional

from .compositions import CompositionModule, composition_resolver
from .init import initializer_resolver, uniform_norm_p1_
from .utils import TransformerEncoder
from .weighting import EdgeWeighting, SymmetricEdgeWeighting, edge_weight_resolver
from ..regularizers import Regularizer, regularizer_resolver
from ..triples import CoreTriplesFactory, TriplesFactory
from ..typing import Constrainer, Hint, HintType, Initializer, Normalizer
from ..utils import Bias, clamp_norm, complex_normalize

__all__ = [
    "RepresentationModule",
    "Embedding",
    "LowRankEmbeddingRepresentation",
    "EmbeddingSpecification",
    "CompGCNLayer",
    "CombinedCompGCNRepresentations",
    "SingleCompGCNRepresentation",
    "LabelBasedTransformerRepresentation",
    "SubsetRepresentationModule",
    # Utils
    "constrainer_resolver",
    "normalizer_resolver",
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

        .. note::

            this method is implemented in subclasses. Prefer using `forward_unique` instead,
            which optimizes for duplicate indices.

        :param indices: shape: s
            The indices, or None. If None, this is interpreted as ``torch.arange(self.max_id)`` (although implemented
            more efficiently).

        :return: shape: (``*s``, ``*self.shape``)
            The representations.
        """

    def forward_unique(
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
        if indices is None:
            return self(None)
        unique, inverse = indices.unique(return_inverse=True)
        return self(unique)[inverse]

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""

    def post_parameter_update(self):
        """Apply constraints which should not be included in gradients."""

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
            # note: this seems to work, as finfo returns the datatype of the underlying floating
            # point dtype, rather than the combined complex one
            dtype = getattr(torch, torch.finfo(dtype).dtype)

        super().__init__(
            max_id=num_embeddings,
            shape=shape,
        )

        # use make for initializer since there's a default, and make_safe
        # for the others to pass through None values
        self.initializer = initializer_resolver.make(initializer, initializer_kwargs)
        self.normalizer = normalizer_resolver.make_safe(normalizer, normalizer_kwargs)
        self.constrainer = constrainer_resolver.make_safe(constrainer, constrainer_kwargs)
        self.regularizer = regularizer_resolver.make_safe(regularizer, regularizer_kwargs)

        self._embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=_embedding_dim,
            dtype=dtype,
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

    def __post_init__(self):
        if self.shape is None:
            if self.embedding_dim is None:
                raise ValueError("Missing both, shape and embedding_dim")
            self.shape = (self.embedding_dim,)

    def make(self, *, num_embeddings: int, device: Optional[torch.device] = None) -> Embedding:
        """Create an embedding with this specification."""
        rv = Embedding(
            num_embeddings=num_embeddings,
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


constrainer_resolver = FunctionResolver([functional.normalize, complex_normalize, torch.clamp, clamp_norm])

normalizer_resolver = FunctionResolver([functional.normalize])


class CompGCNLayer(nn.Module):
    """A single layer of the CompGCN model."""
    # TODO: PyG

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
        :param attention_heads:
            Number of attention heads when using the attention weighting
        :param attention_dropout:
            Dropout for the attention message weighting
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
        self.edge_weighting: EdgeWeighting = edge_weight_resolver.make(
            edge_weighting, output_dim=output_dim, attn_drop=attention_dropout, num_heads=attention_heads
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


class CombinedCompGCNRepresentations(nn.Module):
    """A sequence of CompGCN layers."""
    # TODO: PyG

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
    # TODO: PyG

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
