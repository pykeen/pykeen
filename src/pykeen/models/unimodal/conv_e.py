"""Implementation of ConvE."""

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn.init import xavier_normal_
from ...nn.modules import ConvEInteraction
from ...triples import CoreTriplesFactory
from ...typing import FloatTensor, Hint, Initializer

__all__ = [
    "ConvE",
]

logger = logging.getLogger(__name__)


class ConvE(ERModel[FloatTensor, FloatTensor, tuple[FloatTensor, FloatTensor]]):
    r"""An implementation of ConvE from [dettmers2018]_.

    ConvE represents entities using a $d$-dimensional embedding and a scalar tail bias.
    Relations are represented by a $d$-dimensional vector.
    All three components can be stored as :class:`~pykeen.nn.representation.Embedding`.

    On top of these representations, this model uses the :class:`~pykeen.nn.modules.ConvEInteraction` to calculate
    scores.

    Example::
        .. literalinclude:: ../examples/models/conv_e.py

    ---
    citation:
        author: Dettmers
        year: 2018
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366
        github: TimDettmers/ConvE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        output_channels=dict(type=int, low=4, high=6, scale="power_two"),
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
        output_dropout=DEFAULT_DROPOUT_HPO_RANGE,
        feature_map_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[type[Loss]] = BCEAfterSigmoidLoss  # type: ignore[type-abstract]
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    #: If batch normalization is enabled, this is: num_features – C from an expected input of size (N,C,L)
    bn0: torch.nn.BatchNorm2d | None
    #: If batch normalization is enabled, this is: num_features – C from an expected input of size (N,C,H,W)
    bn1: torch.nn.BatchNorm2d | None
    bn2: torch.nn.BatchNorm1d | None

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        input_channels: int | None = None,
        output_channels: int = 32,
        embedding_height: int | None = None,
        embedding_width: int | None = None,
        kernel_height: int = 3,
        kernel_width: int = 3,
        input_dropout: float = 0.2,
        output_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        embedding_dim: int = 200,
        apply_batch_normalization: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        # ConvE should be trained with inverse triples
        if not triples_factory.create_inverse_triples:
            logger.warning(
                "\nThe ConvE model should be trained with inverse triples.\n"
                "This can be done by defining the TriplesFactory class with the _create_inverse_triples_ parameter set "
                "to true.",
            )

        super().__init__(
            triples_factory=triples_factory,
            interaction=ConvEInteraction,
            interaction_kwargs=dict(
                input_channels=input_channels,
                output_channels=output_channels,
                embedding_height=embedding_height,
                embedding_width=embedding_width,
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                input_dropout=input_dropout,
                output_dropout=output_dropout,
                feature_map_dropout=feature_map_dropout,
                embedding_dim=embedding_dim,
                apply_batch_normalization=apply_batch_normalization,
            ),
            entity_representations_kwargs=[
                # entity embedding
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                ),
                # ConvE uses one bias for each entity
                dict(
                    shape=tuple(),
                    initializer=nn.init.zeros_,
                ),
            ],
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
