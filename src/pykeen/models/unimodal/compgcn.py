# -*- coding: utf-8 -*-

"""Implementation of the Comp-GCN model."""

from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint

from ..nbase import ERModel
from ...nn import EmbeddingSpecification
from ...nn.emb import CompGCNRepresentations, SingleCompGCNRepresentation
from ...nn.modules import DistMultInteraction, Interaction, interaction_resolver
from ...triples import CoreTriplesFactory
from ...typing import RelationRepresentation

__all__ = [
    "CompGCN",
]


class CompGCN(ERModel[torch.FloatTensor, RelationRepresentation, torch.FloatTensor]):
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
        triples_factory: CoreTriplesFactory,
        encoder_kwargs: Optional[Mapping[str, Any]] = None,
        interaction: Hint[Interaction[torch.FloatTensor, RelationRepresentation, torch.FloatTensor]] = None,
        interaction_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the model.

        :param triples_factory:
            The triples factory.
        :param encoder_kwargs:
            Key-word arguments for the encoder, cf. CompGCNRepresentations.
        :param interaction:
            The interaction function to use as decoder.
        :param interaction_kwargs:
            Additional key-word based arguments for the interaction function.
        :param kwargs:
            Additional key-word based arguments passed to ERModel.
        """
        encoder_kwargs = {} if encoder_kwargs is None else dict(encoder_kwargs)

        # TODO: Remove this, once testing is updated
        embedding_dim = kwargs.pop("embedding_dim", None)
        if embedding_dim is not None:
            encoder_kwargs.setdefault("embedding_specification", EmbeddingSpecification(embedding_dim=embedding_dim))

        # combined representation
        combined = CompGCNRepresentations(
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
        if interaction is None:
            interaction = DistMultInteraction
        # TODO: Pass interaction / interaction_kwargs to base class?
        interaction = interaction_resolver.make(query=interaction, pos_kwargs=interaction_kwargs)
        super().__init__(
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            triples_factory=triples_factory,
            interaction=interaction,
            **kwargs,
        )
