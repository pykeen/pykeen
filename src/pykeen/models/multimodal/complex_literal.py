# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model."""

from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.nn as nn

from .base import LiteralInteraction, LiteralModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import ComplExInteraction
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint
from ...utils import combine_complex, split_complex

__all__ = [
    'ComplExLiteral',
]


class ComplExLiteralCombination(nn.Module):
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
        x_im = self.imag(torch.cat([x_im, literal], dim=-1))
        return combine_complex(x_re=x_re, x_im=x_im)


# TODO: Check entire build of the model
# TODO: There are no tests.
class ComplExLiteral(LiteralModel):
    """An implementation of ComplexLiteral from [agustinus2018]_ based on the LCWA training approach."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default = BCEWithLogitsLoss
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
                    dropout=input_dropout,
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
