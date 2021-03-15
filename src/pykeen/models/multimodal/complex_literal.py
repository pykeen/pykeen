# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model."""

from typing import Any, ClassVar, Mapping, Optional, Type

import torch
import torch.nn as nn

from .base import LiteralModel
from .combinations import ComplExLiteralCombination
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import ComplExInteraction, LiteralInteraction
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint

__all__ = [
    'ComplExLiteral',
]


class ComplExLiteral(LiteralModel):
    """An implementation of the LiteralE model with the ComplEx interaction from [kristiadi2018]_.

    ---
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
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.2,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            interaction=LiteralInteraction(
                base=ComplExInteraction(),
                combination=ComplExLiteralCombination(
                    embedding_dim=embedding_dim,
                    num_of_literals=triples_factory.numeric_literals.shape[-1],
                    input_dropout=input_dropout,
                ),
            ),
            entity_specification=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=nn.init.xavier_normal_,
                dtype=torch.complex64,
            ),
            relation_specification=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=nn.init.xavier_normal_,
                dtype=torch.complex64,
            ),
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
