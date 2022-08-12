# -*- coding: utf-8 -*-

"""Various decompositions for R-GCN."""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from class_resolver import ClassResolver, Hint, HintOrType, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from .init import uniform_norm_p1_, xavier_normal_
from .representation import LowRankRepresentation, Representation
from .utils import ShapeError, adjacency_tensor_to_stacked_matrix, use_horizontal_stacking
from .weighting import EdgeWeighting, edge_weight_resolver
from ..triples import CoreTriplesFactory
from ..utils import ExtraReprMixin, einsum

__all__ = [
    "RGCNRepresentation",
    "Decomposition",
    "BasesDecomposition",
    "BlockDecomposition",
    "decomposition_resolver",
]

logger = logging.getLogger(__name__)


class Decomposition(nn.Module, ExtraReprMixin, ABC):
    r"""Base module for relation-specific message passing.

    A decomposition module implementation offers a way to reduce the number of parameters needed by learning
    independent $d^2$ matrices for each relation. In R-GCN, the two proposed variants are treated as
    hyper-parameters, and for different datasets different decompositions are superior in performance.

    The decomposition module itself does not compute the full matrix from the factors, but rather provides efficient
    means to compute the product of the factorized matrix with the source nodes' latent features to construct the
    messages. This is usually more efficient than constructing the full matrices.

    For an intuition, you can think about a simple low-rank matrix factorization of rank `1`, where $W = w w^T$
    for a $d$-dimensional vector `w`. Then, computing $Wv$ as $(w w^T) v$ gives you an intermediate result of size
    $d \times d$, while you can also compute $w(w^Tv)$, where the intermediate result is just a scalar.

    The implementations use the efficient version based on adjacency tensor stacking from [thanapalasingam2021]_.
    The adjacency tensor is reshaped into a sparse matrix to support message passing by a
    single sparse matrix multiplication, cf. :func:`pykeen.nn.utils.adjacency_tensor_to_stacked_matrix`.

    .. note ::
        this module does neither take care of the self-loop, nor of applying an activation function.
    """

    def __init__(
        self,
        num_relations: int,
        input_dim: int = 32,
        output_dim: Optional[int] = None,
    ):
        """Initialize the layer.

        :param num_relations: >0
            The number of relations.
        :param input_dim: >0
            The input dimension.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        """
        super().__init__()
        # input normalization
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.num_relations = num_relations
        self.output_dim = output_dim

    def iter_extra_repr(self) -> Iterable[str]:
        """Iterate over components for `extra_repr`."""
        yield from super().iter_extra_repr()
        yield f"input_dim={self.input_dim}"
        yield f"output_dim={self.output_dim}"
        yield f"num_relations={self.num_relations}"

    def forward(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
        accumulator: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Relation-specific message passing from source to target.

        :param x: shape: (num_nodes, input_dim)
            The node representations.
        :param source: shape: (num_edges,)
            The source indices.
        :param target: shape: (num_edges,)
            The target indices.
        :param edge_type: shape: (num_edges,)
            The edge types.
        :param edge_weights: shape: (num_edges,)
            Precomputed edge weights.
        :param accumulator: shape: (num_nodes, output_dim)
            a pre-allocated output accumulator. may be used if multiple different message passing steps are performed
            and accumulated by sum. If none is given, create an accumulator filled with zeroes.

        :return: shape: (num_nodes, output_dim)
            The enriched node embeddings.
        """
        horizontal = use_horizontal_stacking(input_dim=self.input_dim, output_dim=self.output_dim)
        adj = adjacency_tensor_to_stacked_matrix(
            num_relations=self.num_relations,
            num_entities=x.shape[0],
            source=source,
            target=target,
            edge_type=edge_type,
            edge_weights=edge_weights,
            horizontal=horizontal,
        )
        if horizontal:
            x = self.forward_horizontally_stacked(x=x, adj=adj)
        else:
            x = self.forward_vertically_stacked(x=x, adj=adj)
        if accumulator is not None:
            x = accumulator + x
        return x

    @abstractmethod
    def forward_horizontally_stacked(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for horizontally stacked adjacency.

        :param x: shape: `(num_entities, input_dim)`
            the input entity representations
        :param adj: shape: `(num_entities, num_relations * num_entities)`, sparse
            the horizontally stacked adjacency matrix

        :return: shape: `(num_entities, output_dim)`
            the updated entity representations.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_vertically_stacked(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vertically stacked adjacency.

        :param x: shape: `(num_entities, input_dim)`
            the input entity representations
        :param adj: shape: `(num_entities * num_relations, num_entities)`, sparse
            the vertically stacked adjacency matrix

        :return: shape: `(num_entities, output_dim)`
            the updated entity representations.
        """
        raise NotImplementedError

    def reset_parameters(self):
        """Reset the layer's parameters."""
        # note: the base class does not have any parameters


class BasesDecomposition(Decomposition):
    r"""
    Represent relation-weights as a linear combination of base transformation matrices.

    The basis decomposition represents the relation-specific transformation matrices
    as a weighted combination of base matrices, $\{\mathbf{B}_i^l\}_{i=1}^{B}$, i.e.,

    .. math::
        \mathbf{W}_r^l = \sum \limits_{b=1}^B \alpha_{rb} \mathbf{B}^l_i

    The implementation uses a reshaping of the adjacency tensor into a sparse matrix to support message passing by a
    single sparse matrix multiplication, cf. [thanapalasingam2021]_.

    .. seealso ::
        https://github.com/thiviyanT/torch-rgcn/blob/267faffd09a441d902c483a8c130410c72910e90/torch_rgcn/layers.py#L450-L565
    """

    def __init__(self, num_bases: Optional[int] = None, **kwargs):
        """
        Initialize the bases decomposition.

        :param num_bases:
            the number of bases
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Decomposition.__init__`
        """
        super().__init__(**kwargs)

        # Heuristic for default value
        if num_bases is None:
            num_bases = math.ceil(math.sqrt(self.num_relations))
            logger.info(f"No num_bases was provided. Using sqrt(num_relations)={num_bases}.")

        if num_bases > self.num_relations:
            logger.warning(f"The number of bases ({num_bases}) exceeds the number of relations ({self.num_relations}).")

        self.relation_representations = LowRankRepresentation(
            max_id=self.num_relations,
            shape=(self.input_dim, self.output_dim),
            num_bases=num_bases,
            weight_initializer=uniform_norm_p1_,
            initializer=nn.init.xavier_normal_,
        )

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"num_bases={self.relation_representations.num_bases}"

    @property
    def bases(self) -> torch.Tensor:
        """Return the base representations."""
        return self.relation_representations.bases(indices=None)

    @property
    def base_weights(self) -> torch.Tensor:
        """Return the base weights."""
        return self.relation_representations.weight

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        # note: the only parameters are inside the relation representation module, which has its own reset_parameters
        pass

    # docstr-coverage: inherited
    def forward_horizontally_stacked(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = einsum("ni, rb, bio -> rno", x, self.base_weights, self.bases)
        # TODO: can we change the dimension order to make this contiguous?
        return torch.spmm(adj, x.reshape(-1, self.output_dim))

    # docstr-coverage: inherited
    def forward_vertically_stacked(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = torch.spmm(adj, x)
        x = x.view(self.num_relations, -1, self.input_dim)
        return einsum("rb, bio, rni -> no", self.base_weights, self.bases, x)


def _make_dim_divisible(dim: int, divisor: int, name: str) -> int:
    dim_div, remainder = divmod(dim, divisor)
    if remainder:
        logger.warning(f"{name}={dim} not divisible by {divisor}.")
    dim = dim_div * divisor
    assert dim % divisor == 0
    return dim


def _pad_if_necessary(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply padding if necessary."""
    padding_dim = dim - x.shape[-1]
    if padding_dim < 0:
        raise ValueError("Cannot have a negative padding")
    if padding_dim == 0:
        return x
    return torch.cat([x, x.new_zeros(*x.shape[:-1], padding_dim)], dim=-1)


def _unpad_if_necessary(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Remove padding if necessary."""
    padding_dim = dim - x.shape[-1]
    if padding_dim < 0:
        raise ValueError("Cannot have a negative padding")
    if padding_dim == 0:
        return x
    return x[..., :-padding_dim]


class BlockDecomposition(Decomposition):
    r"""
    Represent relation-specific weight matrices via block-diagonal matrices.

    The block-diagonal decomposition restricts each transformation matrix to a block-diagonal-matrix, i.e.,

    .. math::

        \mathbf{W}_r^l = diag(\mathbf{B}_{r,1}^l, \ldots, \mathbf{B}_{r,B}^l)

    where $\mathbf{B}_{r,i} \in \mathbb{R}^{(d^{(l) }/ B) \times (d^{(l)} / B)}$.

    The implementation is based on the efficient version of [thanapalasingam2021]_, which uses a reshaping of the
    adjacency tensor into a sparse matrix to support message passing by a single sparse matrix multiplication.

    .. seealso ::
        https://github.com/thiviyanT/torch-rgcn/blob/267faffd09a441d902c483a8c130410c72910e90/torch_rgcn/layers.py#L450-L565
    """

    def __init__(self, num_blocks: Optional[int] = None, **kwargs):
        """
        Initialize the layer.

        :param num_blocks:
            the number of blocks.
        :param kwargs:
            keyword-based parameters passed to :meth:`Decomposition.__init__`.
        """
        super().__init__(**kwargs)

        # normalize num blocks
        if num_blocks is None:
            num_blocks = math.gcd(self.input_dim, self.output_dim)
            logger.info(f"Inferred num_blocks={num_blocks} by GCD heuristic.")
        self.num_blocks = num_blocks

        # determine necessary padding
        self.padded_input_dim = _make_dim_divisible(dim=self.input_dim, divisor=num_blocks, name="input_dim")
        self.padded_output_dim = _make_dim_divisible(dim=self.output_dim, divisor=num_blocks, name="output_dim")

        # determine block sizes
        self.input_block_size = self.padded_input_dim // num_blocks
        self.output_block_size = self.padded_output_dim // num_blocks

        # (R, nb, bsi, bso)
        self.blocks = nn.Parameter(
            data=torch.empty(
                self.num_relations,
                num_blocks,
                self.input_block_size,
                self.output_block_size,
            ),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer's parameters."""
        xavier_normal_(self.blocks.data)

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"num_blocks={self.num_blocks}"
        if self.input_block_size == self.output_block_size:
            yield f"block_size={self.input_block_size}"
        else:
            yield f"input_block_size={self.input_block_size}"
            yield f"output_block_size={self.output_block_size}"

    # docstr-coverage: inherited
    def forward_horizontally_stacked(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # apply padding if necessary
        x = _pad_if_necessary(x=x, dim=self.padded_input_dim)
        # (n, di) -> (n, nb, bsi)
        x = x.view(x.shape[0], self.num_blocks, self.input_block_size)
        # (n, nb, bsi), (R, nb, bsi, bso) -> (R, n, nb, bso)
        x = einsum("nbi, rbio -> rnbo", x, self.blocks)
        # (R, n, nb, bso) -> (R * n, do)
        # note: depending on the contracting order, the output may supporting viewing, or not
        x = x.reshape(-1, self.num_blocks * self.output_block_size)
        # (n, R * n), (R * n, do) -> (n, do)
        x = torch.sparse.mm(adj, x)
        # remove padding if necessary
        return _unpad_if_necessary(x=x, dim=self.padded_output_dim)

    # docstr-coverage: inherited
    def forward_vertically_stacked(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # apply padding if necessary
        x = _pad_if_necessary(x=x, dim=self.padded_input_dim)
        # (R * n, n), (n, di) -> (R * n, di)
        x = torch.sparse.mm(adj, x)
        # (R * n, di) -> (R, n, nb, bsi)
        x = x.view(self.num_relations, -1, self.num_blocks, self.input_block_size)
        # (R, nb, bsi, bso), (R, n, nb, bsi) -> (n, nb, bso)
        x = einsum("rbio, rnbi -> nbo", self.blocks, x)
        # (n, nb, bso) -> (n, do)
        # note: depending on the contracting order, the output may supporting viewing, or not
        x = x.reshape(x.shape[0], self.num_blocks * self.output_block_size)
        # remove padding if necessary
        return _unpad_if_necessary(x=x, dim=self.padded_output_dim)


class RGCNLayer(nn.Module):
    r"""
    An RGCN layer from [schlichtkrull2018]_ updated to match the official implementation.

    This layer uses separate decompositions for forward and backward edges (i.e., "normal" and implicitly created
    inverse relations), as well as a separate transformation for self-loops.

    Ignoring dropouts, decomposition and normalization, it can be written as

    .. math ::
        y_i = \sigma(
            W^s x_i
            + \sum_{(e_j, r, e_i) \in \mathcal{T}} W^f_r x_j
            + \sum_{(e_i, r, e_j) \in \mathcal{T}} W^b_r x_j
            + b
        )

    where $b, W^s, W^f_r, W^b_r$ are trainable weights. $W^f_r, W^b_r$ are relation-specific, and commonly enmploy a
    weight-sharing mechanism, cf. Decomposition. $\sigma$ is an activation function. The individual terms in both sums
    are typically weighted. This is implemented by EdgeWeighting. Moreover, RGCN employs an edge-dropout, however,
    this needs to be done outside of an individual layer, since the same edges are dropped across all layers. In
    contrast, the self-loop dropout is layer-specific.
    """

    def __init__(
        self,
        num_relations: int,
        input_dim: int = 32,
        output_dim: Optional[int] = None,
        use_bias: bool = True,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        self_loop_dropout: float = 0.2,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the layer.

        :param input_dim: >0
            the input dimension
        :param num_relations:
            the number of relations
        :param output_dim: >0
            the output dimension. If none is given, use the input dimension.
        :param use_bias:
            whether to use a trainable bias
        :param activation:
            the activation function to use. Defaults to None, i.e., the identity function serves as activation.
        :param activation_kwargs:
            additional keyword-based arguments passed to the activation function for instantiation
        :param self_loop_dropout: 0 <= self_loop_dropout <= 1
            the dropout to use for self-loops
        :param decomposition:
            the decomposition to use, cf. Decomposition and decomposition_resolver
        :param decomposition_kwargs:
            the keyword-based arguments passed to the decomposition for instantiation
        """
        super().__init__()
        # cf. https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/message_gcns/gcn_basis.py#L22-L24  # noqa: E501
        # there are separate decompositions for forward and backward relations.
        # the self-loop weight is not decomposed.
        self.fwd = decomposition_resolver.make(
            query=decomposition,
            pos_kwargs=decomposition_kwargs,
            input_dim=input_dim,
            output_dim=output_dim,
            num_relations=num_relations,
        )
        self.bwd = decomposition_resolver.make(
            query=decomposition,
            pos_kwargs=decomposition_kwargs,
            input_dim=input_dim,
            output_dim=output_dim,
            num_relations=num_relations,
        )
        self.self_loop = nn.Linear(in_features=input_dim, out_features=self.fwd.output_dim, bias=use_bias)
        self.dropout = nn.Dropout(p=self_loop_dropout)
        self.activation = activation_resolver.make_safe(query=activation, pos_kwargs=activation_kwargs)

    def forward(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Calculate enriched entity representations.

        :param x: shape: (num_entities, input_dim)
            The input entity representations.
        :param source: shape: (num_triples,)
            The indices of the source entity per triple.
        :param target: shape: (num_triples,)
            The indices of the target entity per triple.
        :param edge_type: shape: (num_triples,)
            The relation type per triple.
        :param edge_weights: shape: (num_triples,)
            Scalar edge weights per triple.

        :return: shape: (num_entities, output_dim)
            Enriched entity representations.
        """
        # TODO: we could cache the stacked adjacency matrices
        # self-loop
        y = self.dropout(self.self_loop(x))
        # forward messages
        y = self.fwd(x=x, source=source, target=target, edge_type=edge_type, edge_weights=edge_weights, accumulator=y)
        # backward messages
        y = self.bwd(x=x, source=target, target=source, edge_type=edge_type, edge_weights=edge_weights, accumulator=y)
        # activation
        if self.activation is not None:
            y = self.activation(y)
        return y


decomposition_resolver: ClassResolver[Decomposition] = ClassResolver.from_subclasses(
    base=Decomposition, default=BasesDecomposition
)


class RGCNRepresentation(Representation):
    r"""Entity representations enriched by R-GCN.

    The GCN employed by the entity encoder is adapted to include typed edges.
    The forward pass of the GCN is defined by:

     .. math::

        \textbf{e}_{i}^{l+1} = \sigma \left( \sum_{r \in \mathcal{R}}\sum_{j\in \mathcal{N}_{i}^{r}}
        \frac{1}{c_{i,r}} \textbf{W}_{r}^{l} \textbf{e}_{j}^{l} + \textbf{W}_{0}^{l} \textbf{e}_{i}^{l}\right)

    where $\mathcal{N}_{i}^{r}$ is the set of neighbors of node $i$ that are connected to
    $i$ by relation $r$, $c_{i,r}$ is a fixed normalization constant (but it can also be introduced as an additional
    parameter), and $\textbf{W}_{r}^{l} \in \mathbb{R}^{d^{(l)} \times d^{(l)}}$ and
    $\textbf{W}_{0}^{l} \in \mathbb{R}^{d^{(l)} \times d^{(l)}}$ are weight matrices of the `l`-th layer of the
    R-GCN.

    The encoder aggregates for each node $e_i$ the latent representations of its neighbors and its
    own latent representation $e_{i}^{l}$ into a new latent representation $e_{i}^{l+1}$.
    In contrast to standard GCN, R-GCN defines relation specific transformations
    $\textbf{W}_{r}^{l}$ which depend on the type and direction of an edge.

    Since having one matrix for each relation introduces a large number of additional parameters, the authors instead
    propose to use a decomposition, cf. :class:`pykeen.nn.message_passing.Decomposition`.
    """

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        max_id: Optional[int] = None,
        shape: Optional[Sequence[int]] = None,
        entity_representations: HintOrType[Representation] = None,
        entity_representations_kwargs: OptionalKwargs = None,
        num_layers: int = 2,
        use_bias: bool = True,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Hint[EdgeWeighting] = None,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
        cache: bool = True,
        **kwargs,
    ):
        """Instantiate the R-GCN encoder.

        :param triples_factory:
            The triples factory holding the training triples used for message passing.
        :param max_id:
            The maximum number of IDs. could either be None (the default), or match the triples factory's number of
            entities.
        :param shape:
            the shape information. If None, will propagate the shape information of the base entity representations.
        :param entity_representations:
            the base entity representations (or a hint for them)
        :param entity_representations_kwargs:
            additional keyword-based parameters for the base entity representations
        :param num_layers:
            The number of layers.
        :param use_bias:
            Whether to use a bias.
        :param activation:
            The activation.
        :param activation_kwargs:
            Additional keyword based arguments passed if the activation is not pre-instantiated. Ignored otherwise.
        :param edge_dropout:
            The edge dropout to use. Does not apply to self-loops.
        :param self_loop_dropout:
            The self-loop dropout to use.
        :param edge_weighting:
            The edge weighting mechanism.
        :param decomposition:
            The decomposition, cf. :class:`pykeen.nn.message_passing.Decomposition`.
        :param decomposition_kwargs:
            Additional keyword based arguments passed to the decomposition upon instantiation.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`
        :param cache:
            whether to cache representations

        :raises ValueError: If the triples factory creates inverse triples.
        """
        # input validation
        if max_id and max_id != triples_factory.num_entities:
            raise ValueError(
                f"max_id={max_id} differs from triples_factory.num_entities={triples_factory.num_entities}"
            )
        if triples_factory.create_inverse_triples:
            raise ValueError(
                "RGCN internally creates inverse triples. It thus expects a triples factory without them.",
            )

        # has to be imported now to avoid cyclic imports
        from . import representation_resolver

        base = representation_resolver.make(
            entity_representations,
            max_id=triples_factory.num_entities,
            pos_kwargs=entity_representations_kwargs,
        )
        if len(base.shape) > 1:
            raise ValueError(f"{self.__class__.__name__} requires vector base entity representations.")
        max_id = max_id or triples_factory.num_entities
        if max_id != base.max_id:
            raise ValueError(f"Inconsistent max_id={max_id} vs. base.max_id={base.max_id}")
        shape = ShapeError.verify(shape=base.shape, reference=shape)
        super().__init__(max_id=max_id, shape=shape, **kwargs)

        # has to be assigned after call to nn.Module init
        self.entity_embeddings = base

        # Resolve edge weighting
        self.edge_weighting = edge_weight_resolver.make(query=edge_weighting)

        # dropout
        self.edge_dropout = edge_dropout
        self_loop_dropout = self_loop_dropout or edge_dropout

        # Save graph using buffers, such that the tensors are moved together with the model
        h, r, t = triples_factory.mapped_triples.t()
        self.register_buffer("sources", h)
        self.register_buffer("targets", t)
        self.register_buffer("edge_types", r)

        dim = base.shape[0]
        self.layers = nn.ModuleList(
            RGCNLayer(
                input_dim=dim,
                num_relations=triples_factory.num_relations,
                output_dim=dim,
                use_bias=use_bias,
                # no activation on last layer
                # cf. https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/common/model_builder.py#L275  # noqa: E501
                activation=activation if i < num_layers - 1 else None,
                activation_kwargs=activation_kwargs,
                self_loop_dropout=self_loop_dropout,
                decomposition=decomposition,
                decomposition_kwargs=decomposition_kwargs,
            )
            for i in range(num_layers)
        )

        # buffering of enriched representations
        self.enriched_embeddings = None
        self.cache = cache

    # docstr-coverage: inherited
    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()

        # invalidate enriched embeddings
        self.enriched_embeddings = None

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        self.entity_embeddings.reset_parameters()

        for m in self.layers:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            elif any(p.requires_grad for p in m.parameters()):
                logger.warning("Layers %s has parameters, but no reset_parameters.", m)

    def _real_forward_all(self) -> torch.FloatTensor:
        if self.enriched_embeddings is not None:
            return self.enriched_embeddings

        # Bind fields
        # shape: (num_entities, embedding_dim)
        x = self.entity_embeddings(indices=None)
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

        # fixed edges -> pre-compute weights
        if self.edge_weighting is not None and sources.numel() > 0:
            edge_weights = torch.empty_like(sources, dtype=torch.float32)
            for r in range(edge_types.max().item() + 1):
                mask = edge_types == r
                if mask.any():
                    edge_weights[mask] = self.edge_weighting(sources[mask], targets[mask])
        else:
            edge_weights = None

        for layer in self.layers:
            x = layer(
                x=x,
                source=sources,
                target=targets,
                edge_type=edge_types,
                edge_weights=edge_weights,
            )

        # Cache enriched representations
        if self.cache:
            self.enriched_embeddings = x

        return x

    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Enrich the entity embeddings of the decoder using R-GCN message propagation."""
        x = self._real_forward_all()
        if indices is not None:
            x = x[indices]
        return x
