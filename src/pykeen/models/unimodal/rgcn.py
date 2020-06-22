# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

import logging
from os import path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

import torch
from torch import nn
from torch.nn import functional

from . import ComplEx, DistMult, ERMLP
from .. import EntityEmbeddingModel
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
        edge_weighting: Optional[Callable[[torch.LongTensor, torch.LongTensor], torch.FloatTensor]],
        input_dim: int,
        num_relations: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.edge_weighting = edge_weighting
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
        """
        Relation-specific message passing from source to target.

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

    def _aggregate_messages_with_weighting_(
        self,
        message: torch.FloatTensor,
        accumulator: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
    ) -> torch.FloatTensor:
        # optional message weighting
        if self.edge_weighting is not None:
            message = message * self.edge_weighting(source=source, target_r=target)

        # message aggregation
        accumulator.index_add_(dim=0, index=target, source=message)

        return accumulator


class BasesDecomposition(RelationSpecificMessagePassing):
    """Represent relation-weights as a linear combination of base transformation matrices."""

    def __init__(
        self,
        edge_weighting: Optional[Callable[[torch.LongTensor, torch.LongTensor], torch.FloatTensor]],
        input_dim: int,
        num_relations: int,
        num_bases: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__(
            edge_weighting=edge_weighting,
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        # weights
        self.bases = nn.Parameter(
            torch.empty(
                num_bases,
                self.input_dim,
                self.output_dim,
            ), requires_grad=True)
        self.att = nn.Parameter(
            torch.empty(
                num_relations + 1,
                num_bases,
            ), requires_grad=True)

    def reset_parameters(self):  # noqa: D102
        nn.init.xavier_normal_(self.bases)
        # Random convex-combination of bases for initialization (guarantees that initial weight matrices are
        # initialized properly)
        # We have one additional relation for self-loops
        nn.init.uniform_(self.att)
        functional.normalize(self.att.data, p=1, dim=1, out=self.att.data)

    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Transform with all bases, shape: (num_nodes, num_bases, output_dim)
        batch_size = x.shape[0]
        num_bases = self.bases.shape[0]
        t = (x.view(batch_size, 1, 1, -1) @ self.bases).view(batch_size, num_bases, self.output_dim)

        # self-loops first
        if node_keep_mask is not None:
            out = torch.zeros_like(x)
            out[node_keep_mask] = (t[node_keep_mask, :, :] * self.att[None, self.num_relations, :, None]).sum(dim=1)
        else:
            out = (t * self.att[None, self.num_relations, :, None]).sum(dim=1)

        # other relations
        for r in range(self.num_relations):
            # mask, shape: (num_edges,)
            edge_mask = edge_type == r

            if not edge_mask.any():
                # skip relations without edges
                continue

            source_r = source[edge_mask]
            target_r = target[edge_mask]

            # bi-directional message passing
            source_r, target_r = torch.cat([source_r, target_r]), torch.cat([target_r, source_r])

            # compute message, shape: (num_edges_of_type, output_dim)
            m = (t.index_select(dim=0, index=source_r) * self.att[None, r, :, None]).sum(dim=1)

            # optional message weighting
            if edge_weights is not None:
                m = m * edge_weights[edge_mask].unsqueeze(dim=0)

            # message aggregation
            out.index_add_(dim=0, index=target_r, source=m)

        return out


class BlockDecomposition(RelationSpecificMessagePassing):
    """Represent relation-specific weight matrices via block-diagonal matrices."""

    def __init__(
        self,
        edge_weighting: Optional[Callable[[torch.LongTensor, torch.LongTensor], torch.FloatTensor]],
        input_dim: int,
        num_relations: int,
        num_blocks: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__(
            edge_weighting=edge_weighting,
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )
        block_size, remainder = divmod(input_dim, num_blocks)
        if remainder != 0:
            raise NotImplementedError
        self.blocks = nn.Parameter(
            data=torch.empty(
                num_relations + 1,
                num_blocks,
                block_size,
                block_size,
            ), requires_grad=True)

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
        # self-loop first
        start = 0
        out = torch.zeros_like(x)
        for block in self.blocks[-1]:
            stop = start + block.shape[0]
            if node_keep_mask is not None:
                out[node_keep_mask, start:stop] = x[node_keep_mask, start:stop] @ block
            else:
                out[:, start:stop] = x[:, start:stop] @ block

        # other relations
        for r in range(self.num_relations):
            # mask, shape: (num_edges,)
            edge_mask = edge_type == r

            if not edge_mask.any():
                # skip relations without edges
                continue

            source_r = source[edge_mask]
            target_r = target[edge_mask]

            # bi-directional message passing
            source_r, target_r = torch.cat([source_r, target_r]), torch.cat([target_r, source_r])

            # compute message, shape: (num_edges_of_type, output_dim)
            m = []
            start = 0
            for block in self.blocks[r]:
                stop = start + block.shape[0]
                m.append(x[:, start:stop].index_select(dim=0, index=source_r) @ block)
            m = torch.cat(m, dim=-1)

            # optional message weighting
            if edge_weights is not None:
                m = m * edge_weights[edge_mask].unsqueeze(dim=0)

            # message aggregation
            out.index_add_(dim=0, index=target_r, source=m)

        return out


class Bias(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(dim, ), requires_grad=True)

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x + self.bias.unsqueeze(dim=0)


class RGCN(Model):
    """An implementation of R-GCN from [schlichtkrull2018]_.

    This model uses graph convolutions with relation-specific weights.

    .. seealso::

       - `Pytorch Geometric's implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL's implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    """

    #: Interaction model used as decoder
    base_model: EntityEmbeddingModel

    #: The layers
    layers: Sequence[nn.Module]

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=1000, q=50),
        num_bases_or_blocks=dict(type=int, low=2, high=20, q=1),
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
        decomposition=dict(type='categorical', choices=['basis', 'block']),
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
        sparse_messages_owa: bool = True,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Callable[
            [torch.LongTensor, torch.LongTensor],
            torch.FloatTensor
        ] = inverse_indegree_edge_weights,
        decomposition: str = 'basis',
        buffer_messages: bool = True,
    ):
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        if self.triples_factory.create_inverse_triples:
            raise ValueError('R-GCN handles edges in an undirected manner.')

        if base_model is None:
            # Instantiate model
            base_model = DistMult(
                triples_factory=triples_factory,
                embedding_dim=embedding_dim,
                automatic_memory_optimization=automatic_memory_optimization,
                loss=loss,
                preferred_device=preferred_device,
                random_seed=random_seed,
            )
        self.base_model = base_model
        self.base_embeddings = nn.Parameter(
            data=torch.empty(
                self.triples_factory.num_entities,
                embedding_dim,
                device=self.device,
            ),
            requires_grad=True,
        )

        self.embedding_dim = embedding_dim

        # TODO: Fix
        edge_weighting = None
        #: TODO: use enum, of class names
        if decomposition == 'basis':
            if num_bases is None:
                logging.info('Using a heuristic to determine the number of bases.')
                num_bases = triples_factory.num_relations // 2 + 1
            if num_bases > triples_factory.num_relations:
                raise ValueError('The number of bases should not exceed the number of relations.')
        elif decomposition == 'block':
            if num_blocks is None:
                logging.info('Using a heuristic to determine the number of blocks.')
                num_blocks = 2
            if embedding_dim % num_blocks != 0:
                raise ValueError(
                    'With block decomposition, the embedding dimension has to be divisible by the number of'
                    f' blocks, but {embedding_dim} % {num_blocks} != 0.'
                )
        else:
            raise ValueError(decomposition)

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
        self.sparse_messages_owa = sparse_messages_owa

        # Save graph using buffers, such that the tensors are moved together with the model
        h, r, t = self.triples_factory.mapped_triples.t()
        self.register_buffer('sources', h)
        self.register_buffer('targets', t)
        self.register_buffer('edge_types', r)

        layers = []
        for _ in range(num_layers):
            if decomposition == 'basis':
                assert num_bases is not None
                layers.append(BasesDecomposition(
                    edge_weighting=edge_weighting,
                    input_dim=self.embedding_dim,
                    num_relations=self.num_relations,
                    num_bases=num_bases,
                ))
            elif decomposition == 'block':
                assert num_blocks is not None
                layers.append(BlockDecomposition(
                    edge_weighting=edge_weighting,
                    input_dim=self.embedding_dim,
                    num_relations=self.num_relations,
                    num_blocks=num_blocks,
                ))
            else:
                raise AssertionError
            if self.use_bias:
                layers.append(Bias(self.embedding_dim))
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

    def _reset_parameters_(self):
        self.base_model.reset_parameters_()

        # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
        nn.init.xavier_uniform_(self.base_embeddings)

        for m in self.layers:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            elif any(p.requires_grad for p in m.parameters()):
                logger.warning('Layers %s has parameters, but no reset_parameters.', m)

    def _enrich_embeddings(self, batch: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Enrich the entity embeddings using R-GCN message propagation.

        :return: shape: (num_entities, embedding_dim)
            The updated entity embeddings
        """
        # use buffered messages if applicable
        if batch is None and self.enriched_embeddings is not None:
            return self.enriched_embeddings

        # Bind fields
        # shape: (num_entities, embedding_dim)
        x = self.base_embeddings
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
                edge_weights[mask] = self.edge_weighting(source=sources[mask], target=targets[mask])
        else:
            edge_weights = None

        # If batch is given, compute (num_layers)-hop neighbourhood
        if batch is not None:
            start_nodes = torch.cat([batch[:, 0], batch[:, 2]], dim=0)
            edge_mask = _get_neighborhood(
                start_nodes=start_nodes,
                sources=sources,
                targets=targets,
                k=self.num_layers,
                num_nodes=self.num_entities,
                undirected=True,
            )
            sources = sources[edge_mask]
            targets = targets[edge_mask]
            edge_types = edge_types[edge_mask]
            if edge_weights is not None:
                edge_weights = edge_weights[edge_mask]

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

        if batch is None and self.buffer_messages:
            self.enriched_embeddings = x

        return x

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Enrich embeddings
        self.base_model.entity_embeddings.weight.data = self._enrich_embeddings(batch=None)
        return self.base_model.score_hrt(hrt_batch=hrt_batch)
