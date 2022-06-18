# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

from typing import Any, ClassVar, Mapping, Type

import torch.nn as nn

from .base import LiteralModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import ConcatProjectionCombination
from ...nn.modules import DistMultInteraction, Interaction
from ...triples import TriplesNumericLiteralsFactory

__all__ = [
    "DistMultLiteral",
]


class DistMultLiteral(LiteralModel):
    """An implementation of the LiteralE model with the DistMult interaction from [kristiadi2018]_.

    ---
    name: DistMult Literal
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
    interaction_cls: ClassVar[Type[Interaction]] = DistMultInteraction

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory
        :param embedding_dim:
            the embedding dimension
        :param input_dropout:
            the input dropout, cf. :meth:`DistMultCombination.__init__`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`LiteralModel.__init__`
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=self.interaction_cls,
            combination=ConcatProjectionCombination,
            combination_kwargs=dict(
                input_dims=[embedding_dim, triples_factory.literal_shape[0]],
                output_dim=embedding_dim,
                bias=True,
                dropout=input_dropout,
                # no activation
                activation=nn.Identity,
                activation_kwargs=None,
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
