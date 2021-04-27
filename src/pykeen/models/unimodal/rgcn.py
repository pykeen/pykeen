# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint
from torch import nn

from ..nbase import ERModel, EmbeddingSpecificationHint
from ...nn.emb import EmbeddingSpecification, RGCNRepresentations
from ...nn.message_passing import Decomposition
from ...nn.modules import Interaction, interaction_resolver
from ...nn.weighting import EdgeWeighting
from ...triples import CoreTriplesFactory
from ...typing import Initializer, RelationRepresentation

__all__ = [
    "RGCN",
]


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
    ---
    citation:
        author: Schlichtkrull
        year: 2018
        link: https://arxiv.org/pdf/1703.06103
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
        triples_factory: CoreTriplesFactory,
        embedding_dim: int = 500,
        num_layers: int = 2,
        # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
        base_entity_initializer: Hint[Initializer] = nn.init.xavier_uniform_,
        base_entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = nn.init.xavier_uniform_,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
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
        )

        # Resolve interaction function
        interaction = interaction_resolver.make(query=interaction, pos_kwargs=interaction_kwargs)

        # set default relation representation
        if relation_representations is None:
            relation_representations = EmbeddingSpecification(
                shape=entity_representations.shape,
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            )
        super().__init__(
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            triples_factory=triples_factory,
            interaction=interaction,
            **kwargs,
        )
