# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteralGated model."""

from typing import Any, ClassVar, Mapping

import torch.nn as nn

from .base import LiteralModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.combinations import DistMultGatedCombination
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import DistMultInteraction, LiteralInteraction
from ...triples import TriplesNumericLiteralsFactory

__all__ = [
    'DistMultLiteralGated',
]


class DistMultLiteralGated(LiteralModel):
    """An implementation of the LiteralE model with the Gated DistMult interaction from [kristiadi2018]_.
    ---
    name: Gated DistMult Literal
    citation:
        author: Kristiadi
        year: 2018
        link: https://arxiv.org/abs/1802.00934
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(margin=0.0)

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=LiteralInteraction(
                base=DistMultInteraction(),
                combination=DistMultGatedCombination(
                    entity_embedding_dim=embedding_dim,
                    literal_embedding_dim=triples_factory.numeric_literals.shape[1],
                ),
            ),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                ),
            ],
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                ),
            ],
            **kwargs,
        )
