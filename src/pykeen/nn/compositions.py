"""Composition modules."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar

import torch
from class_resolver import ClassResolver
from torch import nn

from ..typing import FloatTensor
from ..utils import circular_correlation

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


Composition = Callable[[FloatTensor, FloatTensor], FloatTensor]


class CompositionModule(nn.Module, ABC):
    """An (element-wise) composition function for vectors."""

    @abstractmethod
    def forward(self, a: FloatTensor, b: FloatTensor) -> FloatTensor:
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

    #: The stateless function that gets composed
    func: ClassVar[Composition]

    # docstr-coverage: inherited
    def forward(self, a: FloatTensor, b: FloatTensor) -> FloatTensor:  # noqa: D102
        return self.__class__.func(a, b)


# NOTE: wrapping torch.sub and torch.mul since their docstrings cause an issue...


class SubtractionCompositionModule(FunctionalCompositionModule):
    """Composition by element-wise subtraction."""

    #: Subtracts with :func:`torch.sub`
    func: ClassVar[Composition] = lambda a, b: torch.sub(a, b)


class MultiplicationCompositionModule(FunctionalCompositionModule):
    """Composition by element-wise multiplication."""

    #: Multiplies with :func:`torch.mul`
    func: ClassVar[Composition] = lambda a, b: torch.mul(a, b)


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
