# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

from typing import Optional, TYPE_CHECKING

import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..base import LiteralModel
from ...losses import Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import DistMultInteraction
from ...regularizers import Regularizer
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
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        input_dropout=dict(type=float, low=0, high=1.0),
    )
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(margin=0.0)

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        input_dropout: float = 0.0,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        predict_with_sigmoid: bool = False,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            interaction=DistMultInteraction(),
            combination=nn.Sequential(
                nn.Linear(embedding_dim + triples_factory.numeric_literals.shape[1], embedding_dim),
                nn.Dropout(input_dropout),
            ),
            entity_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
            ),
            relation_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
            ),
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
