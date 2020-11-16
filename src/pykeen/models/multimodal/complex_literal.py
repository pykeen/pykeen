# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model based on the local closed world assumption (LCWA) training approach."""

from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..base import LiteralModel
from ...losses import BCEWithLogitsLoss, Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import ComplExInteraction
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint
from ...utils import combine_complex, split_complex

__all__ = [
    'ComplExLiteral',
]


class ComplexLiteralCombination(nn.Module):
    """Separately transform real and imaginary part."""

    def __init__(
        self,
        embedding_dim: int,
        num_of_literals: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.real = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim + num_of_literals, embedding_dim),
            torch.nn.Tanh(),
        )
        self.imag = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim + num_of_literals, embedding_dim),
            torch.nn.Tanh(),
        )
        self.embedding_dim = embedding_dim

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        x, literal = x[..., :self.embedding_dim], x[..., self.embedding_dim:]
        x_re, x_im = split_complex(x)
        x_re = self.real(torch.cat([x_re, literal], dim=-1))
        x_im = self.real(torch.cat([x_im, literal], dim=-1))
        return combine_complex(x_re=x_re, x_im=x_im)


# TODO: Check entire build of the model
# TODO: There are no tests.
class ComplExLiteral(LiteralModel):
    """An implementation of ComplexLiteral from [agustinus2018]_ based on the LCWA training approach."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        input_dropout={
            'type': float,
            'low': 0.1,
            'high': 0.3,
        },
    )
    #: The default loss function class
    loss_default = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        input_dropout: float = 0.2,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=2 * embedding_dim,  # complex
            interaction=ComplExInteraction(),
            combination=ComplexLiteralCombination(
                embedding_dim=embedding_dim,
                num_of_literals=triples_factory.numeric_literals.shape[-1],
                dropout=input_dropout,
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
        )
