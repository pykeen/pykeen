# -*- coding: utf-8 -*-

"""Base classes for multi-modal models."""

from typing import Optional, TYPE_CHECKING, Tuple

import torch
from torch import nn

from ..base import ERModel
from ...losses import Loss
from ...nn import Embedding, EmbeddingSpecification, Interaction, LiteralRepresentations
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint, HeadRepresentation, RelationRepresentation, Representation, TailRepresentation

__all__ = [
    "LiteralModel",
]

if TYPE_CHECKING:
    from ...typing import Representation  # noqa


class LiteralInteraction(
    Interaction[
        Tuple[Representation, Representation],
        Representation,
        Tuple[Representation, Representation],
    ],
):

    def __init__(
        self,
        base: Interaction[Representation, Representation, Representation],
        combination: nn.Module,
    ):
        super().__init__()
        self.base = base
        self.combination = combination
        self.entity_shape = tuple(self.base.entity_shape) + ("e",)

    def forward(
        self,
        h: Tuple[Representation, Representation],
        r: Representation,
        t: Tuple[Representation, Representation],
    ) -> torch.FloatTensor:
        # combine entity embeddings + literals
        h = torch.cat(h, dim=-1)
        h = self.combination(h.view(-1, h.shape[-1])).view(*h.shape[:-1], -1)  # type: ignore
        t = torch.cat(t, dim=-1)
        t = self.combination(t.view(-1, t.shape[-1])).view(*t.shape[:-1], -1)  # type: ignore
        return self.base(h=h, r=r, t=t)


class LiteralModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """Base class for models with entity literals."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        interaction: LiteralInteraction,
        entity_specification: Optional[EmbeddingSpecification] = None,
        relation_specification: Optional[EmbeddingSpecification] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_representations=[
                # entity embeddings
                Embedding.from_specification(
                    num_embeddings=triples_factory.num_entities,
                    specification=entity_specification,
                ),
                # Entity literals
                LiteralRepresentations(
                    numeric_literals=torch.as_tensor(triples_factory.numeric_literals, dtype=torch.float32),
                ),
            ],
            relation_representations=Embedding.from_specification(
                num_embeddings=triples_factory.num_relations,
                specification=relation_specification,
            ),
        )
