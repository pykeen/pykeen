# -*- coding: utf-8 -*-

"""Implementation of the Comp-GCN model."""

from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint

from ..nbase import ERModel
from ...nn import EmbeddingSpecification
from ...nn.emb import CompGCNRepresentation, SingleCompGCNRepresentation
from ...nn.modules import DistMultInteraction, Interaction, interaction_resolver
from ...triples import TriplesFactory
from ...typing import RelationRepresentation

__all__ = [
    "CompGCN",
]


class CompGCN(
    ERModel[torch.FloatTensor, RelationRepresentation, torch.FloatTensor],
):
    """An implementation of CompGCN from [vashishth2020]_.

    This model uses graph convolutions, and composition functions.

    ---
    citation:
        author: Vashishth
        year: 2020
        link: https://arxiv.org/pdf/1911.03082
        github: malllabiisc/CompGCN
    """

    #: The default strategy for optimizing the model"s hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=32, high=512, q=32),
    )

    def __init__(
        self,
        *,
        triples_factory: TriplesFactory,
        encoder_kwargs: Optional[Mapping[str, Any]] = None,
        interaction: Hint[Interaction[torch.FloatTensor, RelationRepresentation, torch.FloatTensor]] = DistMultInteraction,
        interaction_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        encoder_kwargs = encoder_kwargs or {}

        # TODO: Remove this, once testing is updated
        embedding_dim = kwargs.pop("embedding_dim")
        if embedding_dim is not None:
            encoder_kwargs.setdefault("embedding_specification", EmbeddingSpecification(embedding_dim=embedding_dim))

        # combined representation
        combined = CompGCNRepresentation(
            triples_factory=triples_factory,
            **(encoder_kwargs or {}),
        )
        # wrap
        entity_representations = SingleCompGCNRepresentation(
            combined=combined,
            position=0,
        )
        relation_representations = SingleCompGCNRepresentation(
            combined=combined,
            position=1,
        )

        # Resolve interaction function
        interaction = interaction_resolver.make(query=interaction, pos_kwargs=interaction_kwargs)
        super().__init__(
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            triples_factory=triples_factory,
            interaction=interaction,
            **kwargs,
        )
