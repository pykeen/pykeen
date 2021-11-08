"""A wrapper which combines an interaction function with NodePiece entity representations."""

from typing import Any, Mapping, Optional

from class_resolver.api import HintOrType

from ..nbase import ERModel
from ...nn.emb import EmbeddingSpecification, NodePieceRepresentation
from ...nn.modules import Interaction, TransEInteraction
from ...triples.triples_factory import CoreTriplesFactory


class NodePieceModel(ERModel):
    """A wrapper which combines an interaction function with NodePiece entity representations."""

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_representations: Optional[EmbeddingSpecification] = None,
        interaction: HintOrType[Interaction] = TransEInteraction,
        node_piece_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory used for tokenization
        :param embedding_specification:
            the embedding specification. Defaults to 64 dimensional embeddings with default settings otherwise.
        :param relation_representations:
            the relation representations. Defaults to embedding_specification.
        :param interaction:
            the interaction module, or a hint for it.
        :param node_piece_kwargs:
            additional keyword-based arguments passed to the NodePieceRepresentation
        :param kwargs:
            additional keyword-based arguments passed to ERModel.__init__
        """
        embedding_specification = embedding_specification or EmbeddingSpecification(
            shape=(64,),
        )
        entity_representations = NodePieceRepresentation(
            triples_factory=triples_factory,
            token_representation=embedding_specification,
            **(node_piece_kwargs or {}),
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=entity_representations,
            relation_representations=relation_representations or embedding_specification,
            **kwargs,
        )
