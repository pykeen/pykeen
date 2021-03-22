# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

import logging
from os import path
from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint, Resolver
from torch import nn

from .decompositions import Decomposition, decomposition_resolver
from .weightings import EdgeWeighting, edge_weight_resolver
from ...nbase import ERModel, EmbeddingSpecificationHint
from ....nn import EmbeddingSpecification, RepresentationModule
from ....nn.modules import Interaction, interaction_resolver
from ....triples import TriplesFactory
from ....typing import Initializer, RelationRepresentation

__all__ = [
    "RGCN",
]

logger = logging.getLogger(name=path.basename(__file__))


class Bias(nn.Module):
    """A module wrapper for adding a bias."""

    def __init__(self, dim: int):
        """Initialize the module.

        :param dim: >0
            The dimension of the input.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.empty(dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer"s parameters."""
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


# TODO: Move to utils
activation_resolver = Resolver.from_subclasses(base=nn.Module, default=nn.ReLU)


class RGCNRepresentations(RepresentationModule):
    """Entity representations enriched by R-GCN."""

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_specification: EmbeddingSpecificationHint,
        num_layers: int = 2,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Hint[EdgeWeighting] = None,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
        buffer_messages: bool = True,
    ):
        base_embeddings = embedding_specification.make(num_embeddings=triples_factory.num_entities)
        super().__init__(max_id=triples_factory.num_entities, shape=base_embeddings.shape)
        self.entity_embeddings = base_embeddings

        # Resolve edge weighting
        self.edge_weighting = edge_weight_resolver.make(query=edge_weighting)

        # dropout
        self.edge_dropout = edge_dropout
        self.self_loop_dropout = self_loop_dropout or edge_dropout

        # batch norm and bias
        use_batch_norm = use_batch_norm
        if use_batch_norm:
            if use_bias:
                logger.warning("Disabling bias because batch normalization is used.")
            use_bias = False

        # Save graph using buffers, such that the tensors are moved together with the model
        h, r, t = triples_factory.mapped_triples.t()
        self.register_buffer("sources", h)
        self.register_buffer("targets", t)
        self.register_buffer("edge_types", r)

        layers = []
        for _ in range(num_layers):
            layers.append(
                decomposition_resolver.make(
                    query=decomposition,
                    pos_kwargs=decomposition_kwargs,
                    input_dim=base_embeddings.embedding_dim,
                    num_relations=triples_factory.num_relations,
                )
            )
            if use_bias:
                layers.append(Bias(dim=base_embeddings.embedding_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=base_embeddings.embedding_dim))
            layers.append(activation_resolver.make(query=activation, pos_kwargs=activation_kwargs))
        self.layers = nn.ModuleList(layers)

        # buffering of messages
        self.buffer_messages = buffer_messages
        self.enriched_embeddings = None

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()

        # invalidate enriched embeddings
        self.enriched_embeddings = None

    def reset_parameters(self):  # noqa: D102
        self.entity_embeddings.reset_parameters()

        for m in self.layers:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            elif any(p.requires_grad for p in m.parameters()):
                logger.warning("Layers %s has parameters, but no reset_parameters.", m)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Enrich the entity embeddings of the decoder using R-GCN message propagation."""
        if self.enriched_embeddings is not None:
            x = self.enriched_embeddings
            if indices is not None:
                x = x[indices]
            return x

        # clear cached embeddings as soon as possible to avoid unnecessary memory consumption
        self.enriched_embeddings = None

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

        # Different dropout for self-loops (only in training mode)
        if self.training and self.self_loop_dropout is not None:
            node_keep_mask = torch.rand(x.shape[0], device=x.device) > self.self_loop_dropout
        else:
            node_keep_mask = None

        # fixed edges -> pre-compute weights
        if self.edge_weighting is not None:
            edge_weights = torch.empty_like(sources, dtype=torch.float32)
            for r in range(edge_types.max().item() + 1):
                mask = edge_types == r
                if mask.any():
                    edge_weights[mask] = self.edge_weighting(sources[mask], targets[mask])
        else:
            edge_weights = None

        for layer in self.layers:
            if isinstance(layer, Decomposition):
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

        # Cache enriched representations
        self.enriched_embeddings = x

        if indices is not None:
            x = x[indices]

        return x


class RGCN(
    ERModel[torch.FloatTensor, RelationRepresentation, torch.FloatTensor],
):
    """An implementation of R-GCN from [schlichtkrull2018]_.

    This model uses graph convolutions with relation-specific weights.

    .. seealso::

       - `Pytorch Geometric"s implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL"s implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    """

    #: The default strategy for optimizing the model"s hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=32, high=512, q=32),
        num_layers=dict(type=int, low=1, high=5, q=1),
        use_bias=dict(type="bool"),
        use_batch_norm=dict(type="bool"),
        activation_cls=dict(type="categorical", choices=[nn.ReLU, nn.LeakyReLU]),
        interaction=dict(type="categorical", choices=["distmult", "complex", "ermlp"]),
        edge_dropout=dict(type=float, low=0.0, high=.9),
        self_loop_dropout=dict(type=float, low=0.0, high=.9),
        edge_weighting=dict(type="categorical", choices=["inverse_in_degree", "inverse_out_degree", "symmetric"]),
        decomposition=dict(type="categorical", choices=["bases", "blocks"]),
        # TODO: Decomposition kwargs
        # num_bases=dict(type=int, low=2, high=100, q=1),
        # num_blocks=dict(type=int, low=2, high=20, q=1),
    )

    def __init__(
        self,
        *,
        triples_factory: TriplesFactory,
        embedding_dim: int = 500,
        num_layers: int = 2,
        # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
        base_entity_initializer: Hint[Initializer] = nn.init.xavier_uniform_,
        base_entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_representations: EmbeddingSpecificationHint = None,
        interaction: Interaction[torch.FloatTensor, RelationRepresentation, torch.FloatTensor],
        interaction_kwargs: Optional[Mapping[str, Any]] = None,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Hint[EdgeWeighting] = None,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
        buffer_messages: bool = True,
        **kwargs,
    ):
        # create enriched entity representations
        entity_representations = RGCNRepresentations(
            triples_factory=triples_factory,
            embedding_specification=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=base_entity_initializer,
                initializer_kwargs=base_entity_initializer_kwargs,
            ),
            num_layers=num_layers,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            activation=activation,
            activation_kwargs=activation_kwargs,
            edge_dropout=edge_dropout,
            self_loop_dropout=self_loop_dropout,
            edge_weighting=edge_weighting,
            decomposition=decomposition,
            decomposition_kwargs=decomposition_kwargs,
            buffer_messages=buffer_messages,
        )

        # Resolve interaction function
        interaction = interaction_resolver.make(query=interaction, pos_kwargs=interaction_kwargs)

        # set default relation representation
        if relation_representations is None:
            relation_representations = EmbeddingSpecification(
                shape=entity_representations.shape,
                initializer=nn.init.xavier_uniform_,
            )
        super().__init__(
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            triples_factory=triples_factory,
            interaction=interaction,
            **kwargs,
        )
