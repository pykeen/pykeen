# -*- coding: utf-8 -*-

"""Various decompositions for R-GCN."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from class_resolver import ClassResolver, Hint, HintOrType, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from .init import uniform_norm_p1_
from .representation import LowRankRepresentation, Representation
from .weighting import EdgeWeighting, edge_weight_resolver
from ..triples import CoreTriplesFactory

__all__ = [
    "RGCNRepresentation",
    "Decomposition",
    "BasesDecomposition",
    "BlockDecomposition",
    "decomposition_resolver",
]

logger = logging.getLogger(__name__)


def _reduce_relation_specific(
    relation: int,
    source: torch.LongTensor,
    target: torch.LongTensor,
    edge_type: torch.LongTensor,
    edge_weights: Optional[torch.FloatTensor],
) -> Union[Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.FloatTensor]], Tuple[None, None, None]]:
    """Reduce edge information to one relation.

    :param relation:
        The relation ID.
    :param source: shape: (num_edges,)
        The source node IDs.
    :param target: shape: (num_edges,)
        The target node IDs.
    :param edge_type: shape: (num_edges,)
        The edge types.
    :param edge_weights: shape: (num_edges,)
        The edge weights.

    :return:
        The source, target, weights for edges related to the desired relation type.
    """
    # mask, shape: (num_edges,)
    edge_mask = edge_type == relation
    if not edge_mask.any():
        return None, None, None

    source_r = source[edge_mask]
    target_r = target[edge_mask]
    if edge_weights is not None:
        edge_weights = edge_weights[edge_mask]

    # bi-directional message passing
    source_r, target_r = torch.cat([source_r, target_r]), torch.cat([target_r, source_r])
    if edge_weights is not None:
        edge_weights = torch.cat([edge_weights, edge_weights])

    return source_r, target_r, edge_weights


class Decomposition(nn.Module, ABC):
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
    """

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        output_dim: Optional[int] = None,
    ):
        """Initialize the layer.

        :param input_dim: >0
            The input dimension.
        :param num_relations: >0
            The number of relations.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        """Reset the parameters of this layer."""
        raise NotImplementedError


class BasesDecomposition(Decomposition):
    r"""Represent relation-weights as a linear combination of base transformation matrices.

    The basis decomposition represents the relation-specific transformation matrices
    as a weighted combination of base matrices, $\{\mathbf{B}_i^l\}_{i=1}^{B}$, i.e.,

    .. math::

        \mathbf{W}_r^l = \sum \limits_{b=1}^B \alpha_{rb} \mathbf{B}^l_i
    """

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        output_dim: Optional[int] = None,
        memory_intense: bool = False,
    ):
        """Initialize the layer.

        :param input_dim: >0
            The input dimension.
        :param num_relations: >0
            The number of relations.
        :param num_bases: >0
            The number of bases to use.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        :param memory_intense:
            Enable memory-intense forward pass which may be faster, in particular if the number of different relations
            is small.
        :raises ValueError: If the ``num_bases`` is greater than ``num_relations``
        """
        super().__init__(
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        # Heuristic for default value
        if num_bases is None:
            logger.info("No num_bases was provided. Falling back to 2.")
            num_bases = 2

        if num_bases > num_relations:
            raise ValueError("The number of bases should not exceed the number of relations.")

        self.relation_representations = LowRankRepresentation(
            max_id=num_relations,
            shape=(self.input_dim, self.output_dim),
            weight_initializer=uniform_norm_p1_,
            initializer=nn.init.xavier_normal_,
        )
        self.memory_intense = memory_intense

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        self.relation_representations.reset_parameters()

    def _get_weight(self, relation_id: int) -> torch.FloatTensor:
        """Construct weight matrix for a specific relation ID.

        :param relation_id:
            The relation ID.

        :return:
            A 2-D matrix.
        """
        return self.relation_representations(indices=torch.as_tensor(relation_id)).squeeze(dim=0)

    def _forward_memory_intense(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        out: torch.FloatTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # other relations
        m = torch.einsum(
            "mi,mij->mj",
            x.index_select(dim=0, index=source),
            self.relation_representations(indices=edge_type),
        )
        if edge_weights is not None:
            m = m * edge_weights.unsqueeze(dim=-1)
        return out.index_add(dim=0, index=target, source=m)

    def _forward_memory_light(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        out: torch.FloatTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # other relations
        for r in range(self.num_relations):
            # Select source and target indices as well as edge weights for the
            # currently considered relation
            source_r, target_r, weights_r = _reduce_relation_specific(
                relation=r,
                source=source,
                target=target,
                edge_type=edge_type,
                edge_weights=edge_weights,
            )

            # skip relations without edges
            if source_r is None:
                continue

            # compute message, shape: (num_edges_of_type, output_dim)
            w = self._get_weight(relation_id=r)
            # since we may have one node ID appearing multiple times as source
            # ID, we can save some computation by first reducing to the unique
            # source IDs, compute transformed representations and afterwards
            # select these representations for the correct edges.
            uniq_source_r, inv_source_r = source_r.unique(return_inverse=True)
            # select unique source node representations
            m = x[uniq_source_r]
            # transform representations by relation specific weight
            m = m @ w
            # select the uniquely transformed representations for each edge
            m = m.index_select(dim=0, index=inv_source_r)

            # optional message weighting
            if weights_r is not None:
                m = m * weights_r.unsqueeze(dim=-1)

            # message aggregation
            out = out.index_add(dim=0, index=target_r, source=m)

        return out

    # docstr-coverage: inherited
    def forward(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
        accumulator: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if accumulator is None:
            accumulator = torch.zeros_like(x)
        if self.memory_intense:
            _forward = self._forward_memory_intense
        else:
            _forward = self._forward_memory_light
        return _forward(
            x=x,
            source=source,
            target=target,
            edge_type=edge_type,
            out=accumulator,
            edge_weights=edge_weights,
        )


class BlockDecomposition(Decomposition):
    r"""Represent relation-specific weight matrices via block-diagonal matrices.

    The block-diagonal decomposition restricts each transformation matrix to a block-diagonal-matrix, i.e.,

    .. math::

        \mathbf{W}_r^l = diag(\mathbf{B}_{r,1}^l, \ldots, \mathbf{B}_{r,B}^l)

    where $\mathbf{B}_{r,i} \in \mathbb{R}^{(d^{(l) }/ B) \times (d^{(l)} / B)}$.
    """

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        num_blocks: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        """Initialize the layer.

        :param input_dim: >0
            The input dimension.
        :param num_relations: >0
            The number of relations.
        :param num_blocks: >0
            The number of blocks to use. Has to be a divisor of input_dim.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        :raises NotImplementedError: If ``input_dim`` is not divisible by ``num_blocks``
        """
        super().__init__(
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        if num_blocks is None:
            logger.info("Using a heuristic to determine the number of blocks.")
            num_blocks = min(i for i in range(2, input_dim + 1) if input_dim % i == 0)

        block_size, remainder = divmod(input_dim, num_blocks)
        if remainder != 0:
            raise NotImplementedError(
                "With block decomposition, the embedding dimension has to be divisible by the number of"
                f" blocks, but {input_dim} % {num_blocks} != 0.",
            )

        self.blocks = nn.Parameter(
            data=torch.empty(
                num_relations + 1,
                num_blocks,
                block_size,
                block_size,
            ),
            requires_grad=True,
        )
        self.num_blocks = num_blocks
        self.block_size = block_size

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        block_size = self.blocks.shape[-1]
        # Xavier Glorot initialization of each block
        std = torch.sqrt(torch.as_tensor(2.0)) / (2 * block_size)
        nn.init.normal_(self.blocks, std=std)

    # docstr-coverage: inherited
    def forward(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
        accumulator: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # accumulator
        if accumulator is None:
            accumulator = torch.zeros_like(x)

        # view as blocks
        x = x.view(-1, self.num_blocks, self.block_size)
        accumulator = accumulator.view(-1, self.num_blocks, self.block_size)

        # other relations
        for r in range(self.num_relations):
            source_r, target_r, weights_r = _reduce_relation_specific(
                relation=r,
                source=source,
                target=target,
                edge_type=edge_type,
                edge_weights=edge_weights,
            )

            # skip relations without edges
            if source_r is None:
                continue

            # compute message, shape: (num_edges_of_type, num_blocks, block_size)
            uniq_source_r, inv_source_r = source_r.unique(return_inverse=True)
            w_r = self.blocks[r]
            m = torch.einsum("nbi,bij->nbj", x[uniq_source_r], w_r).index_select(dim=0, index=inv_source_r)

            # optional message weighting
            if weights_r is not None:
                m = m * weights_r.unsqueeze(dim=1).unsqueeze(dim=2)

            # message aggregation
            accumulator.index_add_(dim=0, index=target_r, source=m)

        return accumulator.view(-1, self.output_dim)


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
        input_dim: int,
        num_relations: int,
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
            num_relations=num_relations,
        )
        output_dim = self.fwd.output_dim
        self.bwd = decomposition_resolver.make(
            query=decomposition,
            pos_kwargs=decomposition_kwargs,
            input_dim=input_dim,
            num_relations=num_relations,
        )
        self.w_self_loop = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(output_dim)) if use_bias else None
        self.dropout = nn.Dropout(p=self_loop_dropout)
        if activation is not None:
            activation = activation_resolver.make(query=activation, pos_kwargs=activation_kwargs)
        self.activation = activation

    # docstr-coverage: inherited
    def reset_parameters(self):  # noqa: D102
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.xavier_normal_(self.w_self_loop)

    def forward(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ):
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
        # self-loop
        y = self.dropout(x @ self.w_self_loop)
        # forward messages
        y = self.fwd(
            x=x,
            source=source,
            target=target,
            edge_type=edge_type,
            edge_weights=edge_weights,
            accumulator=y,
        )
        # backward messages
        y = self.bwd(
            x=x,
            source=target,
            target=source,
            edge_type=edge_type,
            edge_weights=edge_weights,
            accumulator=y,
        )
        if self.bias is not None:
            y = y + self.bias
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
        :raises ValueError: If the triples factory creates inverse triples.
        """
        if max_id:
            assert max_id == triples_factory.num_entities

        # has to be imported now to avoid cyclic imports
        from . import representation_resolver

        base_embeddings = representation_resolver.make(
            entity_representations,
            max_id=triples_factory.num_entities,
            pos_kwargs=entity_representations_kwargs,
        )
        super().__init__(max_id=base_embeddings.max_id, shape=shape or base_embeddings.shape, **kwargs)
        self.entity_embeddings = base_embeddings

        if triples_factory.create_inverse_triples:
            raise ValueError(
                "RGCN internally creates inverse triples. It thus expects a triples factory without them.",
            )

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

        dim = base_embeddings.embedding_dim
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
