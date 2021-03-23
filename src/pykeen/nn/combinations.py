# -*- coding: utf-8 -*-

"""Implementation of combinations for the LiteralE model."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..utils import combine_complex, split_complex

__all__ = [
    'Combination',
    'DistMultCombination',
    'ComplexCombination',
    'ComplExLiteralCombination',
]


class Combination(nn.Module, ABC):
    """Base class for combinations."""

    @abstractmethod
    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the embedding and literal then score."""


class DistMultCombination(Combination):
    """Transform the embeddings and the literals together."""

    def __init__(
        self,
        embedding_dim: int,
        num_of_literals: int,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        linear = nn.Linear(embedding_dim + num_of_literals, embedding_dim)
        dropout = nn.Dropout(input_dropout)
        self.sequential = nn.Sequential(linear, dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.sequential(x)


class ComplexCombination(Combination):
    """A generalized combination for models using complex tensors."""

    def __init__(
        self,
        real: nn.Module,
        imag: nn.Module,
    ):
        super().__init__()
        self.real = real
        self.imag = imag

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply the real and imaginary sequences separately, then recombine."""
        x, literal = x[..., :self.embedding_dim], x[..., self.embedding_dim:]
        x_re, x_im = split_complex(x)
        x_re = self.real(torch.cat([x_re, literal], dim=-1))
        x_im = self.imag(torch.cat([x_im, literal], dim=-1))
        return combine_complex(x_re=x_re, x_im=x_im)


class ComplExLiteralCombination(ComplexCombination):
    """Separately transform real and imaginary part."""

    def __init__(
        self,
        embedding_dim: int,
        num_of_literals: int,
        input_dropout: float = 0.0,
    ):
        real = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(embedding_dim + num_of_literals, embedding_dim),
            torch.nn.Tanh(),
        )
        imag = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(embedding_dim + num_of_literals, embedding_dim),
            torch.nn.Tanh(),
        )
        super().__init__(real=real, imag=imag)
