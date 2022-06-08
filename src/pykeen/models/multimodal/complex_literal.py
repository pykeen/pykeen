# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model."""

from typing import Any, ClassVar, Mapping, Type

import torch
import torch.nn as nn

from .base import LiteralModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn import ComplexSeparatedCombination, ConcatProjectionCombination
from ...nn.modules import ComplExInteraction, Interaction
from ...triples import TriplesNumericLiteralsFactory

__all__ = [
    "ComplExLiteral",
]


class ComplExLiteral(LiteralModel):
    """An implementation of the LiteralE model with the ComplEx interaction from [kristiadi2018]_.

    This module is a configuration of the general :class:`pykeen.models.LiteralModel` with the
    :class:`pykeen.nn.modules.ComplExInteraction` and :class:`pykeen.nn.combinations.ComplExLiteralCombination`.
    ---
    name: ComplEx Literal
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
    interaction_cls: ClassVar[Type[Interaction]] = ComplExInteraction

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            interaction=self.interaction_cls,
            entity_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                    dtype=torch.complex64,
                ),
            ],
            relation_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                    dtype=torch.complex64,
                ),
            ],
            combination=ComplexSeparatedCombination,
            combination_kwargs=dict(
                # the individual combination for real/complex parts
                combination=ConcatProjectionCombination,
                combination_kwargs=dict(
                    input_dims=[embedding_dim, triples_factory.literal_shape[0]],
                    output_dim=embedding_dim,
                    bias=True,
                    dropout=input_dropout,
                    activation=nn.Tanh,
                    activation_kwargs=None,
                ),
            ),
            **kwargs,
        )
