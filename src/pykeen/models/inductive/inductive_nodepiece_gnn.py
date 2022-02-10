# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Iterable, Mapping, Optional, Tuple, cast

import torch
from torch import nn

from .inductive_nodepiece import InductiveNodePiece
from ...nn.emb import CompGCNLayer
from ...triples import CoreTriplesFactory
from ...typing import (
    TESTING,
    TRAINING,
    VALIDATION,
    HeadRepresentation,
    InductiveMode,
    MappedTriples,
    RelationRepresentation,
    TailRepresentation,
)

__all__ = [
    "InductiveNodePieceGNN",
]

logger = logging.getLogger(__name__)


class BufferedGraph(nn.Module):
    """A pair of edge index and edge type buffers."""

    edge_index: torch.LongTensor
    edge_type: torch.LongTensor

    def __init__(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> None:
        """Create instance."""
        super().__init__()
        self.register_buffer(name="edge_index", tensor=edge_index)
        self.register_buffer(name="edge_type", tensor=edge_type)

    @classmethod
    def from_triples(cls, mapped_triples: MappedTriples) -> "BufferedGraph":
        """Create from mapped triples."""
        return cls(
            edge_index=mapped_triples[:, [0, 2]].t(),
            edge_type=mapped_triples[:, 1],
        )


class InductiveNodePieceGNN(InductiveNodePiece):
    """Inductive NodePiece with a GNN encoder on top.

    Overall, it's a 3-step procedure:

    1. Featurizing nodes via NodePiece
    2. Message passing over the active graph using NodePiece features
    3. Scoring function for a given batch of triples

    As of now, message passing is expected to be over the full graph
    """

    graphs: Mapping[InductiveMode, BufferedGraph]

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        inference_factory: CoreTriplesFactory,
        validation_factory: Optional[CoreTriplesFactory] = None,
        test_factory: Optional[CoreTriplesFactory] = None,
        gnn_encoder: Optional[Iterable[nn.Module]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory. Must have create_inverse_triples set to True.
        :param gnn_encoder:
            an interable of message passing layers. Defaults to 2-layer CompGCN with Hadamard composition.
        :param kwargs:
            additional keyword-based parameters passed to `InductiveNodePiece.__init__`.
        """
        super().__init__(
            triples_factory=triples_factory,
            inference_factory=inference_factory,
            validation_factory=validation_factory,
            test_factory=test_factory,
            **kwargs,
        )

        if gnn_encoder is None:
            # default composition is DistMult-style
            dim = self.entity_representations[0].shape[0]
            gnn_encoder = [
                CompGCNLayer(
                    input_dim=dim,
                    output_dim=dim,
                    activation=torch.nn.ReLU,
                    dropout=0.1,
                )
                for _ in range(2)
            ]
        self.gnn_encoder = nn.ModuleList(gnn_encoder)

        # Saving edge indices for all the supplied splits
        self.graphs = nn.ModuleDict(
            {
                mode: BufferedGraph.from_triples(mapped_triples=factory.mapped_triples)
                for mode, factory in (
                    (None, triples_factory),
                    (TRAINING, inference_factory),
                    (VALIDATION, inference_factory),
                    (TESTING, inference_factory),
                )
                if factory is not None
            }
        )

    def reset_parameters_(self):
        """Reset the GNN encoder explicitly in addition to other params."""
        super().reset_parameters_()
        if getattr(self, "gnn_encoder", None) is not None:
            for layer in self.gnn_encoder:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def _get_representations(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        mode: InductiveMode = None,
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails, in canonical shape with a GNN encoder."""
        entity_representations = self._entity_representation_from_mode(mode=mode)

        # Extract all entity and relation representations
        x_e, x_r = entity_representations[0](), self.relation_representations[0]()

        # Perform message passing and get updated states
        graph = self.graphs[mode]
        for layer in self.gnn_encoder:
            x_e, x_r = layer(
                x_e=x_e,
                x_r=x_r,
                edge_index=graph.edge_index,
                edge_type=graph.edge_type,
            )

        # Use updated entity and relation states to extract requested IDs
        # TODO I got lost in all the Representation Modules and shape casting and wrote this ;(

        hh, rr, tt = [
            x_e.index_select(dim=0, index=h) if h is not None else x_e,
            x_r.index_select(dim=0, index=r) if r is not None else x_r,
            x_e.index_select(dim=0, index=t) if t is not None else x_e,
        ]

        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hh, rr, tt)),
        )
