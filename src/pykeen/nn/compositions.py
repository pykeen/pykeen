# -*- coding: utf-8 -*-

"""Composition modules."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar

import torch
from class_resolver import Resolver
from torch import nn

from .functional import circular_correlation

__all__ = [
    'CompositionModule',
    'FunctionalCompositionModule',
    'SubtractionCompositionModule',
    'MultiplicationCompositionModule',
    'CircularCorrelationCompositionModule',
    'composition_resolver',
]


class CompositionModule(nn.Module, ABC):
    """An (elementwise) composition function for vectors."""

    @abstractmethod
    def forward(self, a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
        """Compose two batches of vectors.

        The tensors have to be broadcastable.

        :param a: shape: s_1
            The first tensor.
        :param b: shape: s_2
            The second tensor.
        :return: shape: s
        """


class FunctionalCompositionModule(CompositionModule):
    """Composition by a function (i.e. state-less)."""

    func: ClassVar[Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]]

    def forward(self, a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.__class__.func(a, b)


class SubtractionCompositionModule(FunctionalCompositionModule):
    """Composition by element-wise subtraction."""

    func = torch.sub


class MultiplicationCompositionModule(FunctionalCompositionModule):
    """Composition by element-wise multiplication."""

    func = torch.mul


class CircularCorrelationCompositionModule(FunctionalCompositionModule):
    """Composition by circular correlation via :func:`pykeen.nn.functional.circular_correlation`."""

    func = circular_correlation


composition_resolver = Resolver.from_subclasses(
    CompositionModule,
    default=MultiplicationCompositionModule,
    skip={
        FunctionalCompositionModule,
    },
)
