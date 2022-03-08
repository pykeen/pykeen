# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteralGated model."""

from typing import Any, ClassVar, Mapping

import torch.nn as nn

from .base import LiteralModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.combinations import GatedCombination
from ...nn.modules import DistMultInteraction, LiteralInteraction
from ...triples import TriplesNumericLiteralsFactory

__all__ = [
    "DistMultLiteralGated",
]


class DistMultLiteralGated(LiteralModel):
    """An implementation of the LiteralE model with thhe Gated DistMult interaction from [kristiadi2018]_.

    This model is different from :class:`pykeen.models.DistMultLiteral` because it uses a
    gate (like found in `LSTMs <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_)
    instead of a LinearDropout module.

    This gate implements the full $g$ function described in the LiteralE paper (see equation 4).
    ---
    name: DistMult Literal (Gated)
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
        input_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=LiteralInteraction(
                base=DistMultInteraction(),
                combination=GatedCombination(
                    entity_embedding_dim=embedding_dim,
                    literal_embedding_dim=triples_factory.numeric_literals.shape[1],
                    input_dropout=input_dropout,
                ),
            ),
            entity_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                ),
            ],
            relation_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                ),
            ],
            **kwargs,
        )
