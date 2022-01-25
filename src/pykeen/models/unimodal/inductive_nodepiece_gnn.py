# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Optional, Tuple

import torch

from .inductive_nodepiece import InductiveNodePiece
from ..nbase import cast
from ...nn.emb import CompGCNLayer
from ...typing import HeadRepresentation, Mode, RelationRepresentation, TailRepresentation

__all__ = [
    "InductiveNodePieceGNN",
]

logger = logging.getLogger(__name__)


class InductiveNodePieceGNN(InductiveNodePiece):
    """Inductive NodePiece with a GNN encoder on top.
    Overall, it's a 3-step procedure:
    1. Featurizing nodes via NodePiece
    2. Message passing over the active graph using NodePiece features
    3. Scoring function for a given batch of triples

    As of now, message passing is expected to be over the full graph
    """

    def __init__(
        self,
        *,
        gnn_encoder: torch.nn.ModuleList = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param gnn_encoder: ModuleList
            ModuleList of message passing layers.
            If not specified, defaults to 2-layer CompGCN with model's embedding dimension and interaction function

        """
        super().__init__(**kwargs)

        train_factory, inference_factory, validation_factory, test_factory = (
            kwargs.get("triples_factory", None),
            kwargs.get("inference_factory", None),
            kwargs.get("validation_factory", None),
            kwargs.get("test_factory", None),
        )

        if gnn_encoder is None:
            # default composition is DistMult-style
            self.gnn_encoder = torch.nn.ModuleList(
                [
                    CompGCNLayer(
                        input_dim=self.entity_representations[0].tokens.shape[0],
                        output_dim=self.entity_representations[0].tokens.shape[0],
                        activation=torch.nn.ReLU,
                        dropout=0.1,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.gnn_encoder = gnn_encoder

        # Saving edge indices for all the supplied splits
        self.register_buffer(name="train_edge_index", tensor=train_factory.mapped_triples[:, [0, 2]].t())
        self.register_buffer(name="train_edge_type", tensor=train_factory.mapped_triples[:, 1])

        if inference_factory is not None:
            inference_edge_index = inference_factory.mapped_triples[:, [0, 2]].t()
            inference_edge_type = inference_factory.mapped_triples[:, 1]

            self.register_buffer(name="valid_edge_index", tensor=inference_edge_index)
            self.register_buffer(name="valid_edge_type", tensor=inference_edge_type)
            self.register_buffer(name="test_edge_index", tensor=inference_edge_index)
            self.register_buffer(name="test_edge_type", tensor=inference_edge_type)
        else:
            self.register_buffer(name="valid_edge_index", tensor=validation_factory.mapped_triples[:, [0, 2]].t())
            self.register_buffer(name="valid_edge_type", tensor=validation_factory.mapped_triples[:, 1])
            self.register_buffer(name="test_edge_index", tensor=test_factory.mapped_triples[:, [0, 2]].t())
            self.register_buffer(name="test_edge_type", tensor=test_factory.mapped_triples[:, 1])

    def _get_representations(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        mode: Mode = None,
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails, in canonical shape with a GNN encoder."""
        entity_representations = self._entity_representation_from_mode(mode=mode)

        # Extract all entity and relation representations
        x_e, x_r = entity_representations[0](), self.relation_representations[0]()

        # Perform message passing and get updated states
        for layer in self.gnn_encoder:
            x_e, x_r = layer(
                x_e=x_e,
                x_r=x_r,
                edge_index=getattr(self, f"{mode}_edge_index"),
                edge_type=getattr(self, f"{mode}_edge_type"),
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
