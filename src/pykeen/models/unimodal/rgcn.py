# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

import logging
from os import path
from typing import Any, Callable, Mapping, Optional, Type

import torch
from torch import nn
from torch.nn import functional

from . import ComplEx, DistMult, ERMLP
from .. import EntityEmbeddingModel
from ..base import Model
from ...losses import Loss
from ...nn import Embedding, RepresentationModule
from ...triples import TriplesFactory
from ...typing import DeviceHint

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
        super().__init__()

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
                    m_r *= self.edge_weighting(source=sources_r, target=targets_r).unsqueeze(dim=-1)

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


class Decoder(nn.Module):
    # TODO: Replace this by interaction function, once https://github.com/pykeen/pykeen/pull/107 is merged.
    def forward(self, h, r, t):
        return (h * r * t).sum(dim=-1)


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

    #: The blocks of the relation-specific weight matrices
    #: shape: (num_relations, num_blocks, embedding_dim//num_blocks, embedding_dim//num_blocks)
    blocks: Optional[nn.ParameterList]

    #: The base weight matrices to generate relation-specific weights
    #: shape: (num_bases, embedding_dim, embedding_dim)
    bases: Optional[nn.ParameterList]

    #: The relation-specific weights for each base
    #: shape: (num_relations, num_bases)
    att: Optional[nn.ParameterList]

    #: The biases for each layer (if used)
    #: shape of each element: (embedding_dim,)
    biases: Optional[nn.ParameterList]

    #: Batch normalization for each layer (if used)
    batch_norms: Optional[nn.ModuleList]

    #: Activations for each layer (if used)
    activations: Optional[nn.ModuleList]

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
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
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
    ):
        if triples_factory.create_inverse_triples:
            raise ValueError('R-GCN handles edges in an undirected manner.')
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.entity_representations = RGCNRepresentations(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            num_bases_or_blocks=num_bases_or_blocks,
            num_layers=num_layers,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            activation_cls=activation_cls,
            activation_kwargs=activation_kwargs,
            sparse_messages_slcwa=sparse_messages_slcwa,
            edge_dropout=edge_dropout,
            self_loop_dropout=self_loop_dropout,
            edge_weighting=edge_weighting,
            decomposition=decomposition,
            buffer_messages=buffer_messages,
            base_representations=None,
        )
        self.relation_embeddings = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
        )
        # TODO: Dummy
        self.decoder = Decoder()

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()
        self.entity_representations.post_parameter_update()
        self.relation_embeddings.post_parameter_update()

    def _reset_parameters_(self):
        self.entity_representations.reset_parameters()
        self.relation_embeddings.reset_parameters()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Enrich embeddings
        h = self.entity_representations(indices=hrt_batch[:, 0])
        t = self.entity_representations(indices=hrt_batch[:, 2])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        return self.decoder(h, r, t).unsqueeze(dim=-1)
