# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

from typing import Optional, TYPE_CHECKING

import torch.nn as nn

from .base import LiteralModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import DistMultInteraction
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint

if TYPE_CHECKING:
    from ...typing import Representation  # noqa

__all__ = [
    'DistMultLiteral',
]


# TODO: Check entire build of the model
# TODO: There are no tests
class DistMultLiteral(LiteralModel):
    """An implementation of DistMultLiteral from [agustinus2018]_."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(margin=0.0)

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.0,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        predict_with_sigmoid: bool = False,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=DistMultInteraction(),
            combination=nn.Sequential(
                nn.Linear(embedding_dim + triples_factory.numeric_literals.shape[1], embedding_dim),
                nn.Dropout(input_dropout),
            ),
            entity_specification=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=nn.init.xavier_normal_,
            ),
            relation_specification=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=nn.init.xavier_normal_,
            ),
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
