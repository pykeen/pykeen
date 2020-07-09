# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

import logging
from os import path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional

from . import ComplEx, DistMult, ERMLP
from .. import EntityRelationEmbeddingModel
from ..base import Model
from ...losses import Loss
from ...triples import TriplesFactory

__all__ = [
    'RGCN',
]

logger = logging.getLogger(name=path.basename(__file__))


def _get_neighborhood(
    start_nodes: torch.LongTensor,
    sources: torch.LongTensor,
    targets: torch.LongTensor,
    k: int,
    num_nodes: int,
    undirected: bool = False,
) -> torch.BoolTensor:
    # Construct node neighbourhood mask
    node_mask = torch.zeros(num_nodes, device=start_nodes.device, dtype=torch.bool)

    # Set nodes in batch to true
    node_mask[start_nodes] = True

    # Compute k-neighbourhood
    for _ in range(k):
        # if the target node needs an embeddings, so does the source node
        node_mask[sources] |= node_mask[targets]

        if undirected:
            node_mask[targets] |= node_mask[sources]

    # Create edge mask
    edge_mask = node_mask[targets]

    if undirected:
        edge_mask |= node_mask[sources]

    return edge_mask


# pylint: disable=unused-argument
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


# pylint: disable=unused-argument
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


class RelationSpecificMessagePassing(nn.Module):
    """Base module for relation-specific message passing."""

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

    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Relation-specific message passing from source to target.

        :param x: shape: (num_nodes, input_dim)
            The node representations.
        :param node_keep_mask: shape: (num_nodes,)
            The node-keep mask for self-loop dropout.
        :param source: shape: (num_edges,)
            The source indices.
        :param target: shape: (num_edges,)
            The target indices.
        :param edge_type: shape: (num_edges,)
            The edge types.
        :param edge_weights: shape: (num_edges,)
            Precomputed edge weights.

        :return: shape: (num_nodes, output_dim)
            The enriched node embeddings.
        """
        raise NotImplementedError

    def reset_parameters(self):
        """Reset the parameters of this layer."""
        raise NotImplementedError


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


class BasesDecomposition(RelationSpecificMessagePassing):
    """Represent relation-weights as a linear combination of base transformation matrices."""

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        num_bases: Optional[int],
        output_dim: Optional[int] = None,
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
        """
        super().__init__(
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        # Heuristic for default value
        if num_bases is None:
            logging.info('No num_bases was provided. Falling back to 2.')
            num_bases = 2

        if num_bases > num_relations:
            raise ValueError('The number of bases should not exceed the number of relations.')

        # weights
        self.bases = nn.Parameter(
            torch.empty(
                num_bases,
                self.input_dim,
                self.output_dim,
            ), requires_grad=True)
        self.relation_base_weights = nn.Parameter(
            torch.empty(
                num_relations + 1,
                num_bases,
            ), requires_grad=True)

    def reset_parameters(self):  # noqa: D102
        nn.init.xavier_normal_(self.bases)
        # Random convex-combination of bases for initialization (guarantees that initial weight matrices are
        # initialized properly)
        # We have one additional relation for self-loops
        nn.init.uniform_(self.relation_base_weights)
        functional.normalize(self.relation_base_weights.data, p=1, dim=1, out=self.relation_base_weights.data)

    def _get_weight(self, relation_id: int) -> torch.FloatTensor:
        """Construct weight matrix for a specific relation ID.

        :param relation_id:
            The relation ID.

        :return:
            A 2-D matrix.
        """
        return torch.einsum('bij,b->ij', self.bases, self.relation_base_weights[relation_id])

    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # TODO: Make this a choice (more memory-intense, but faster)
        # trans_x = torch.einsum('bij,ni->nbj', self.bases, x)

        # self-loops first
        # the last relation_id refers to the self-loop
        w = self._get_weight(relation_id=self.num_relations)
        if node_keep_mask is not None:
            assert node_keep_mask.shape == x.shape[:1]
            out = torch.empty_like(x)
            out[node_keep_mask] = x[node_keep_mask] @ w
            out[~node_keep_mask] = 0.0
        else:
            out = x @ w

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


class BlockDecomposition(RelationSpecificMessagePassing):
    """Represent relation-specific weight matrices via block-diagonal matrices."""

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
        """
        super().__init__(
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        if num_blocks is None:
            logging.info('Using a heuristic to determine the number of blocks.')
            num_blocks = min(i for i in range(2, input_dim + 1) if input_dim % i == 0)

        block_size, remainder = divmod(input_dim, num_blocks)
        if remainder != 0:
            raise NotImplementedError(
                'With block decomposition, the embedding dimension has to be divisible by the number of'
                f' blocks, but {input_dim} % {num_blocks} != 0.'
            )

        self.blocks = nn.Parameter(
            data=torch.empty(
                num_relations + 1,
                num_blocks,
                block_size,
                block_size,
            ), requires_grad=True)
        self.num_blocks = num_blocks
        self.block_size = block_size

    def reset_parameters(self):  # noqa: D102
        block_size = self.blocks.shape[-1]
        # Xavier Glorot initialization of each block
        std = torch.sqrt(torch.as_tensor(2.)) / (2 * block_size)
        nn.init.normal_(self.blocks, std=std)

    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # view as blocks
        x = x.view(-1, self.num_blocks, self.block_size)

        # self-loop first
        out = torch.zeros_like(x)
        w = self.blocks[-1]
        if node_keep_mask is not None:
            out[node_keep_mask] = torch.einsum('nbi,bij->nbj', x[node_keep_mask], w)
        else:
            out = torch.einsum('nbi,bij->nbj', x, w)

        # other relations
        for r in range(self.num_relations):
            specific = _reduce_relation_specific(
                relation=r,
                source=source,
                target=target,
                edge_type=edge_type,
                edge_weights=edge_weights,
            )

            # skip relations without edges
            if specific is None:
                continue

            source_r, target_r, weights_r = specific

            # compute message, shape: (num_edges_of_type, num_blocks, block_size)
            uniq_source_r, inv_source_r = source_r.unique(return_inverse=True)
            w_r = self.blocks[r]
            m = torch.einsum('nbi,bij->nbj', x[uniq_source_r], w_r).index_select(dim=0, index=inv_source_r)

            # optional message weighting
            if weights_r is not None:
                m = m * weights_r.unsqueeze(dim=1).unsqueeze(dim=2)

            # message aggregation
            out.index_add_(dim=0, index=target_r, source=m)

        return out.reshape(-1, self.output_dim)


class Bias(nn.Module):
    """A module wrapper for adding a bias."""

    def __init__(self, dim: int):
        """Initialize the module.

        :param dim: >0
            The dimension of the input.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.empty(dim, ), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer's parameters."""
        nn.init.zeros_(self.bias)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Add the learned bias to the input.

        :param x: shape: (n, d)
            The input.

        :return:
            x + b[None, :]
        """
        return x + self.bias.unsqueeze(dim=0)


class RGCN(EntityRelationEmbeddingModel):
    """An implementation of R-GCN from [schlichtkrull2018]_.

    This model uses graph convolutions with relation-specific weights.

    .. seealso::

       - `Pytorch Geometric's implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL's implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    """

    #: The layers
    layers: Sequence[nn.Module]

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=1000, q=50),
        num_bases=dict(type=int, low=2, high=100, q=1),
        num_blocks=dict(type=int, low=2, high=20, q=1),
        num_layers=dict(type=int, low=1, high=5, q=1),
        use_bias=dict(type='bool'),
        use_batch_norm=dict(type='bool'),
        activation_cls=dict(type='categorical', choices=[None, nn.ReLU, nn.LeakyReLU]),
        base_model_cls=dict(type='categorical', choices=[DistMult, ComplEx, ERMLP]),
        edge_dropout=dict(type=float, low=0.0, high=.9),
        self_loop_dropout=dict(type=float, low=0.0, high=.9),
        edge_weighting=dict(type='categorical', choices=[
            None,
            inverse_indegree_edge_weights,
            inverse_outdegree_edge_weights,
            symmetric_edge_weights,
        ]),
        decomposition=dict(type='categorical', choices=[BasesDecomposition, BlockDecomposition]),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 500,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        num_layers: int = 2,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation_cls: Optional[Type[nn.Module]] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        base_model: Optional[Model] = None,
        sparse_messages_slcwa: bool = True,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Callable[
            [torch.LongTensor, torch.LongTensor],
            torch.FloatTensor
        ] = inverse_indegree_edge_weights,
        decomposition: Type[RelationSpecificMessagePassing] = BasesDecomposition,
        buffer_messages: bool = True,
    ):
        """Initialize the model.

        :param triples_factory:
            The triples factory.
        :param embedding_dim:
            The embedding dimension to use. The same dimension is kept for all message passing layers.
        :param automatic_memory_optimization:
            Whether to apply automatic memory optimization for evaluation.
        :param loss:
            The loss function.
        :param predict_with_sigmoid:
            Whether to apply sigmoid on the model's output in evaluation mode.
        :param preferred_device:
            The preferred device.
        :param random_seed:
            The random seed used for initializing weights.
        :param num_bases: >0
            The number of bases. Requires decomposition=BasesDecomposition to become effective.
        :param num_blocks: >0
            The number of blocks. Requires decomposition=BlockDecomposition to become effective.
        :param num_layers:
            The number of layers.
        :param use_bias:
            Whether to use a bias.
        :param use_batch_norm:
            Whether to use batch normalization layers.
        :param activation_cls:
            The activation function to use.
        :param activation_kwargs:
            Additional key-word based parameters used to instantiate the activation layer.
        :param base_model:
            The base model, i.e. which interaction function to use as a decoder.
        :param sparse_messages_owa:
            Whether to use sparse messages when training with OWA, i.e. do not compute representations for all nodes,
            but only those in a neighborhood of the currently considered ones. Theoretically improves memory
            requirements and runtime, since only a part of the messages are computed, but leads to additional masking.
            Moreover, real-world graphs often exhibit small-world properties leading to neighborhoods quickly comprising
            larger parts of the graph.
        :param edge_dropout:
            The edge dropout to use. Set to None to disable edge dropout.
        :param self_loop_dropout:
            The edge dropout to use for self-loops. Set to None to disable edge dropout.
        :param edge_weighting:
            The edge weighting function to use.
        :param decomposition:
            The decomposition of the relation-specific weight matrices.
        :param buffer_messages:
            Whether to buffer messages. Useful for instance in evaluation mode, when the parameters remain unchanged,
            but many forward passes are requested.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        if self.triples_factory.create_inverse_triples:
            raise ValueError('R-GCN handles edges in an undirected manner.')

        self.embedding_dim = embedding_dim

        # buffering of messages
        self.buffer_messages = buffer_messages
        self.enriched_embeddings = None

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

        layers = []
        message_passing_kwargs = dict(
            input_dim=self.embedding_dim,
            num_relations=self.num_relations,
        )
        if decomposition is BasesDecomposition:
            message_passing_kwargs['num_bases'] = num_bases
        elif decomposition is BlockDecomposition:
            message_passing_kwargs['num_blocks'] = num_blocks
        for _ in range(num_layers):
            layers.append(decomposition(**message_passing_kwargs))
            if self.use_bias:
                layers.append(Bias(dim=self.embedding_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=self.embedding_dim))
            layers.append(self.activation_cls(**(self.activation_kwargs or {})))
        self.layers = nn.ModuleList(layers)

        # Finalize initialization
        self.reset_parameters_()

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()

        # invalidate enriched embeddings
        self.enriched_embeddings = None

    def _reset_parameters_(self):  # noqa: D102
        # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        for m in self.layers:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            elif any(p.requires_grad for p in m.parameters()):
                logger.warning('Layers %s has parameters, but no reset_parameters.', m)

    def _enrich_embeddings_(self) -> torch.FloatTensor:
        """Enrich the entity embeddings of the decoder using R-GCN message propagation."""
        # TODO: Start of commented out for debugging.
        # # use buffered messages if applicable
        # if self.enriched_embeddings is not None:
        #     return self.enriched_embeddings
        #
        # # clear cached embeddings as soon as possible to avoid unnecessary memory consumption
        # self.enriched_embeddings = None
        # TODO: End of commented out for debugging

        # Bind fields
        # shape: (num_entities, embedding_dim)
        x = self.entity_embeddings.weight
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
            node_keep_mask = torch.rand(self.num_entities, device=x.device) > self.self_loop_dropout
        else:
            node_keep_mask = None

        # fixed edges -> pre-compute weights
        if self.edge_weighting is not None:
            edge_weights = torch.empty_like(sources, dtype=torch.float32)
            for r in range(self.num_relations):
                mask = edge_types == r
                if mask.any():
                    edge_weights[mask] = self.edge_weighting(source=sources[mask], target=targets[mask])
        else:
            edge_weights = None

        for layer in self.layers:
            if isinstance(layer, RelationSpecificMessagePassing):
                kwargs = dict(
                    node_keep_mask=node_keep_mask,
                    source=sources,
                    target=targets,
                    edge_type=edge_types,
                    edge_weights=edge_weights,
                )
            else:
                kwargs = dict()
            x = layer(x, **kwargs)

        # TODO: Start of commented out for debugging.
        # # Cache enriched representations
        # self.enriched_embeddings = x
        # TODO: End of commented out for debugging

        return x

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Enrich embeddings
        x = self._enrich_embeddings_()
        h, r, t = hrt_batch.t()
        return torch.einsum('bd,bd,bd->b', x[h], self.relation_embeddings(r), x[t]).unsqueeze(dim=-1)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Enrich embeddings
        x = self._enrich_embeddings_()
        r, t = rt_batch.t()
        return torch.einsum('ed,bd,bd->be', x, self.relation_embeddings(r), x[t])

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Enrich embeddings
        x = self._enrich_embeddings_()
        h, r = hr_batch.t()
        return torch.einsum('bd,bd,ed->be', x[h], self.relation_embeddings(r), x)
