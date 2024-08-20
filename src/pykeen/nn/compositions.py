"""Composition modules."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar

import torch
from class_resolver import ClassResolver
from torch import nn

from .functional import circular_correlation

__all__ = [
    # Base
    "CompositionModule",
    # Concrete
    "FunctionalCompositionModule",
    "SubtractionCompositionModule",
    "MultiplicationCompositionModule",
    "CircularCorrelationCompositionModule",
    # Resolver
    "composition_resolver",
]


Composition = Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]


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
    """Composition by a function (i.e. stateless)."""

    func: ClassVar[Composition]

    # docstr-coverage: inherited
    def forward(self, a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.__class__.func(a, b)


class SubtractionCompositionModule(FunctionalCompositionModule):
    """Composition by elementwise subtraction."""

    func: ClassVar[Composition] = torch.sub


class MultiplicationCompositionModule(FunctionalCompositionModule):
    """Composition by elementwise multiplication."""

    func: ClassVar[Composition] = torch.mul


class CircularCorrelationCompositionModule(FunctionalCompositionModule):
    """Composition by circular correlation via :func:`pykeen.nn.functional.circular_correlation`."""

    func: ClassVar[Composition] = circular_correlation


composition_resolver: ClassResolver[CompositionModule] = ClassResolver.from_subclasses(
    CompositionModule,
    default=MultiplicationCompositionModule,
    skip={
        FunctionalCompositionModule,
    },
)
