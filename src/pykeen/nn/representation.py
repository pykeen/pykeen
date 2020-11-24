# -*- coding: utf-8 -*-

"""Embedding modules."""

import dataclasses
import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy
import torch
import torch.nn
from torch import nn
from torch.nn import functional

from ..regularizers import Regularizer
from ..triples import TriplesFactory
from ..typing import Constrainer, Initializer, Normalizer
from ..utils import upgrade_to_sequence

__all__ = [
    'RepresentationModule',
    'Embedding',
    'EmbeddingSpecification',
    'LiteralRepresentations',
    'RGCNRepresentations',
]

logger = logging.getLogger(__name__)

DIMS = dict(h=1, r=2, t=3)


def _normalize_dim(dim: Union[int, str]) -> int:
    """Normalize the dimension selection."""
    if isinstance(dim, int):
        return dim
    return DIMS[dim.lower()[0]]


def get_expected_canonical_shape(
    indices: Union[None, int, Tuple[int, int], torch.LongTensor],
    dim: Union[str, int],
    suffix_shape: Union[int, Sequence[int]],
    num: Optional[int] = None,
) -> Tuple[int, ...]:
    """
    Calculate the expected canonical shape for the given parameters.

    :param indices:
        The indices, their shape, or None, if no indices are to be provided.
    :param dim:
        The dimension, either symbolic, or numeric.
    :param suffix_shape:
        The suffix-shape.
    :param num:
        The number of representations, if indices_shape is None, i.e. 1-n scoring.

    :return: (batch_size, num_heads, num_relations, num_tails, ``*``).
        The expected shape, a tuple of at least 5 positive integers.
    """
    if torch.is_tensor(indices):
        indices = indices.shape
    exp_shape = [1, 1, 1, 1] + list(suffix_shape)
    dim = _normalize_dim(dim=dim)
    if indices is None:  # 1-n scoring
        exp_shape[dim] = num
    else:  # batch dimension
        exp_shape[0] = indices[0]
        if len(indices) > 1:  # multi-target batching
            exp_shape[dim] = indices[1]
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


def inverse_indegree_edge_weights(source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
    """Normalize messages by inverse in-degree.

    :param source: shape: (num_edges,)
        The source indices.
    :param target: shape: (num_edges,)
        The target indices.

    :return: shape: (num_edges,)
         The edge weights.
    """
    # Calculate in-degree, i.e. number of incoming edges
    uniq, inv, cnt = torch.unique(target, return_counts=True, return_inverse=True)
    return cnt[inv].float().reciprocal()


def inverse_outdegree_edge_weights(source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
    """Normalize messages by inverse out-degree.

    :param source: shape: (num_edges,)
        The source indices.
    :param target: shape: (num_edges,)
        The target indices.

    :return: shape: (num_edges,)
         The edge weights.
    """
    # Calculate in-degree, i.e. number of incoming edges
    uniq, inv, cnt = torch.unique(source, return_counts=True, return_inverse=True)
    return cnt[inv].float().reciprocal()


def symmetric_edge_weights(source: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
    """Normalize messages by product of inverse sqrt of in-degree and out-degree.

    :param source: shape: (num_edges,)
        The source indices.
    :param target: shape: (num_edges,)
        The target indices.

    :return: shape: (num_edges,)
         The edge weights.
    """
    return (
        inverse_indegree_edge_weights(source=source, target=target)
        * inverse_outdegree_edge_weights(source=source, target=target)
    ).sqrt()


class RGCNRepresentations(RepresentationModule):
    """Representations enriched by R-GCN."""

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 500,
        num_bases_or_blocks: int = 5,
        num_layers: int = 2,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation_cls: Optional[Type[nn.Module]] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        sparse_messages_slcwa: bool = True,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Callable[
            [torch.LongTensor, torch.LongTensor],
            torch.FloatTensor,
        ] = inverse_indegree_edge_weights,
        decomposition: str = 'basis',
        buffer_messages: bool = True,
        base_representations: Optional[RepresentationModule] = None,
    ):
        super().__init__(shape=(embedding_dim,), max_id=triples_factory.num_entities)

        self.triples_factory = triples_factory

        # normalize representations
        if base_representations is None:
            base_representations = Embedding(
                num_embeddings=triples_factory.num_entities,
                embedding_dim=embedding_dim,
                # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
                initializer=nn.init.xavier_uniform_,
            )
        self.base_embeddings = base_representations
        self.embedding_dim = embedding_dim

        # check decomposition
        self.decomposition = decomposition
        if self.decomposition == 'basis':
            if num_bases_or_blocks is None:
                logging.info('Using a heuristic to determine the number of bases.')
                num_bases_or_blocks = triples_factory.num_relations // 2 + 1
            if num_bases_or_blocks > triples_factory.num_relations:
                raise ValueError('The number of bases should not exceed the number of relations.')
        elif self.decomposition == 'block':
            if num_bases_or_blocks is None:
                logging.info('Using a heuristic to determine the number of blocks.')
                num_bases_or_blocks = 2
            if embedding_dim % num_bases_or_blocks != 0:
                raise ValueError(
                    'With block decomposition, the embedding dimension has to be divisible by the number of'
                    f' blocks, but {embedding_dim} % {num_bases_or_blocks} != 0.',
                )
        else:
            raise ValueError(f'Unknown decomposition: "{decomposition}". Please use either "basis" or "block".')

        self.num_bases = num_bases_or_blocks
        self.edge_weighting = edge_weighting
        self.edge_dropout = edge_dropout
        if self_loop_dropout is None:
            self_loop_dropout = edge_dropout
        self.self_loop_dropout = self_loop_dropout
        self.use_batch_norm = use_batch_norm
        if activation_cls is None:
            activation_cls = nn.ReLU
        self.activation_cls = activation_cls
        self.activation_kwargs = activation_kwargs
        if use_batch_norm:
            if use_bias:
                logger.warning('Disabling bias because batch normalization was used.')
            use_bias = False
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.sparse_messages_slcwa = sparse_messages_slcwa

        # Save graph using buffers, such that the tensors are moved together with the model
        h, r, t = self.triples_factory.mapped_triples.t()
        self.register_buffer('sources', h)
        self.register_buffer('targets', t)
        self.register_buffer('edge_types', r)

        self.activations = nn.ModuleList([
            self.activation_cls(**(self.activation_kwargs or {})) for _ in range(self.num_layers)
        ])

        # Weights
        self.bases = nn.ParameterList()
        if self.decomposition == 'basis':
            self.att = nn.ParameterList()
            for _ in range(self.num_layers):
                self.bases.append(nn.Parameter(
                    data=torch.empty(
                        self.num_bases,
                        self.embedding_dim,
                        self.embedding_dim,
                    ),
                    requires_grad=True,
                ))
                self.att.append(nn.Parameter(
                    data=torch.empty(
                        self.triples_factory.num_relations + 1,
                        self.num_bases,
                    ),
                    requires_grad=True,
                ))
        elif self.decomposition == 'block':
            block_size = self.embedding_dim // self.num_bases
            for _ in range(self.num_layers):
                self.bases.append(nn.Parameter(
                    data=torch.empty(
                        self.triples_factory.num_relations + 1,
                        self.num_bases,
                        block_size,
                        block_size,
                    ),
                    requires_grad=True,
                ))

            self.att = None
        else:
            raise NotImplementedError
        if self.use_bias:
            self.biases = nn.ParameterList([
                nn.Parameter(torch.empty(self.embedding_dim), requires_grad=True)
                for _ in range(self.num_layers)
            ])
        else:
            self.biases = None
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_features=self.embedding_dim)
                for _ in range(self.num_layers)
            ])
        else:
            self.batch_norms = None

        # buffering of messages
        self.buffer_messages = buffer_messages
        self.enriched_embeddings = None

    def _get_relation_weights(self, i_layer: int, r: int) -> torch.FloatTensor:
        if self.decomposition == 'block':
            # allocate weight
            w = torch.zeros(self.embedding_dim, self.embedding_dim, device=self.bases[i_layer].device)

            # Get blocks
            this_layer_blocks = self.bases[i_layer]

            # self.bases[i_layer].shape (num_relations, num_blocks, embedding_dim/num_blocks, embedding_dim/num_blocks)
            # note: embedding_dim is guaranteed to be divisible by num_bases in the constructor
            block_size = self.embedding_dim // self.num_bases
            for b, start in enumerate(range(0, self.embedding_dim, block_size)):
                stop = start + block_size
                w[start:stop, start:stop] = this_layer_blocks[r, b, :, :]

        elif self.decomposition == 'basis':
            # The current basis weights, shape: (num_bases)
            att = self.att[i_layer][r, :]
            # the current bases, shape: (num_bases, embedding_dim, embedding_dim)
            b = self.bases[i_layer]
            # compute the current relation weights, shape: (embedding_dim, embedding_dim)
            w = torch.sum(att[:, None, None] * b, dim=0)

        else:
            raise AssertionError(f'Unknown decomposition: {self.decomposition}')

        return w

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # use buffered messages if applicable
        if indices is None and self.enriched_embeddings is not None:
            return self.enriched_embeddings
        if indices is not None and indices.ndimension() > 1:
            raise RuntimeError("indices must be None, or 1-dimensional.")

        # Bind fields
        # shape: (num_entities, embedding_dim)
        x = self.base_embeddings(indices=None)
        sources = self.sources
        targets = self.targets
        edge_types = self.edge_types

        # Edge dropout: drop the same edges on all layers (only in training mode)
        if self.training and self.edge_dropout is not None:
            # Get random dropout mask
            edge_keep_mask = torch.rand(self.sources.shape[0], device=x.device) > self.edge_dropout

            # Apply to edges
            sources = sources[edge_keep_mask]
            targets = targets[edge_keep_mask]
            edge_types = edge_types[edge_keep_mask]

        # Different dropout for self-loops (only in training mode)
        if self.training and self.self_loop_dropout is not None:
            node_keep_mask = torch.rand(self.triples_factory.num_entities, device=x.device) > self.self_loop_dropout
        else:
            node_keep_mask = None

        for i in range(self.num_layers):
            # Initialize embeddings in the next layer for all nodes
            new_x = torch.zeros_like(x)

            # TODO: Can we vectorize this loop?
            for r in range(self.triples_factory.num_relations):
                # Choose the edges which are of the specific relation
                mask = (edge_types == r)

                # No edges available? Skip rest of inner loop
                if not mask.any():
                    continue

                # Get source and target node indices
                sources_r = sources[mask]
                targets_r = targets[mask]

                # send messages in both directions
                sources_r, targets_r = torch.cat([sources_r, targets_r]), torch.cat([targets_r, sources_r])

                # Select source node embeddings
                x_s = x[sources_r]

                # get relation weights
                w = self._get_relation_weights(i_layer=i, r=r)

                # Compute message (b x d) * (d x d) = (b x d)
                m_r = x_s @ w

                # Normalize messages by relation-specific in-degree
                if self.edge_weighting is not None:
                    m_r *= self.edge_weighting(sources_r, targets_r).unsqueeze(dim=-1)

                # Aggregate messages in target
                new_x.index_add_(dim=0, index=targets_r, source=m_r)

            # Self-loop
            self_w = self._get_relation_weights(i_layer=i, r=self.triples_factory.num_relations)
            if node_keep_mask is None:
                new_x += x @ self_w
            else:
                new_x[node_keep_mask] += x[node_keep_mask] @ self_w

            # Apply bias, if requested
            if self.use_bias:
                bias = self.biases[i]
                new_x += bias

            # Apply batch normalization, if requested
            if self.use_batch_norm:
                batch_norm = self.batch_norms[i]
                new_x = batch_norm(new_x)

            # Apply non-linearity
            if self.activations is not None:
                activation = self.activations[i]
                new_x = activation(new_x)

            x = new_x

        if indices is None and self.buffer_messages:
            self.enriched_embeddings = x
        if indices is not None:
            x = x[indices]

        return x

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()

        # invalidate enriched embeddings
        self.enriched_embeddings = None

    def reset_parameters(self):
        self.base_embeddings.reset_parameters()

        gain = nn.init.calculate_gain(nonlinearity=self.activation_cls.__name__.lower())
        if self.decomposition == 'basis':
            for base in self.bases:
                nn.init.xavier_normal_(base, gain=gain)
            for att in self.att:
                # Random convex-combination of bases for initialization (guarantees that initial weight matrices are
                # initialized properly)
                # We have one additional relation for self-loops
                nn.init.uniform_(att)
                functional.normalize(att.data, p=1, dim=1, out=att.data)
        elif self.decomposition == 'block':
            for base in self.bases:
                block_size = base.shape[-1]
                # Xavier Glorot initialization of each block
                std = torch.sqrt(torch.as_tensor(2.)) * gain / (2 * block_size)
                nn.init.normal_(base, std=std)

        # Reset biases
        if self.biases is not None:
            for bias in self.biases:
                nn.init.zeros_(bias)

        # Reset batch norm parameters
        if self.batch_norms is not None:
            for bn in self.batch_norms:
                bn.reset_parameters()

        # Reset activation parameters, if any
        for act in self.activations:
            if hasattr(act, 'reset_parameters'):
                act.reset_parameters()
