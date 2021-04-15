# -*- coding: utf-8 -*-

"""Regularization in PyKEEN."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Mapping, Optional

import torch
from class_resolver import Resolver, normalize_string
from torch import nn
from torch.nn import functional

from .utils import lp_norm, powersum_norm

__all__ = [
    # Base Class
    'Regularizer',
    # Child classes
    'LpRegularizer',
    'NoRegularizer',
    'CombinedRegularizer',
    'PowerSumRegularizer',
    'TransHRegularizer',
    # Utils
    'regularizer_resolver',
]

_REGULARIZER_SUFFIX = 'Regularizer'


class Regularizer(nn.Module, ABC):
    """A base class for all regularizers."""

    #: The overall regularization weight
    weight: torch.FloatTensor

    #: The current regularization term (a scalar)
    regularization_term: torch.FloatTensor

    #: Should the regularization only be applied once? This was used for ConvKB and defaults to False.
    apply_only_once: bool

    #: Has this regularizer been updated since last being reset?
    updated: bool

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]

    def __init__(
        self,
        weight: float = 1.0,
        apply_only_once: bool = False,
        parameters: Optional[Iterable[nn.Parameter]] = None,
    ):
        """Instantiate the regularizer.

        :param weight: The relative weight of the regularization
        :param apply_only_once: Should the regularization be applied more than once after reset?
        :param parameters: Specific parameters to track. if none given, it's expected that your
            model automatically delegates to the :func:`update` function.
        """
        super().__init__()
        self.tracked_parameters = list(parameters) if parameters else []
        self.register_buffer(name='weight', tensor=torch.as_tensor(weight))
        self.apply_only_once = apply_only_once
        self.register_buffer(name="regularization_term", tensor=torch.zeros(1, dtype=torch.float))
        self.updated = False
        self.reset()

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the regularizer class."""
        return normalize_string(cls.__name__, suffix=_REGULARIZER_SUFFIX)

    def add_parameter(self, parameter: nn.Parameter) -> None:
        """Add a parameter for regularization."""
        self.tracked_parameters.append(parameter)

    def reset(self) -> None:
        """Reset the regularization term to zero."""
        self.regularization_term.detach_().zero_()
        self.updated = False

    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the regularization term for one tensor."""
        raise NotImplementedError

    def update(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularization term based on passed tensors."""
        if not self.training or not torch.is_grad_enabled() or (self.apply_only_once and self.updated):
            return
        self.regularization_term = self.regularization_term + sum(self.forward(x=x) for x in tensors)
        self.updated = True

    @property
    def term(self) -> torch.FloatTensor:
        """Return the weighted regularization term."""
        return self.regularization_term * self.weight

    def pop_regularization_term(self) -> torch.FloatTensor:
        """Return the weighted regularization term, and reset the regularize afterwards."""
        # If there are tracked parameters, update based on them
        if self.tracked_parameters:
            self.update(*self.tracked_parameters)

        term = self.regularization_term
        self.reset()
        return self.weight * term


class NoRegularizer(Regularizer):
    """A regularizer which does not perform any regularization.

    Used to simplify code.
    """

    #: The default strategy for optimizing the no-op regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}

    def update(self, *tensors: torch.FloatTensor) -> None:  # noqa: D102
        # no need to compute anything
        pass

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        # always return zero
        return torch.zeros(1, dtype=x.dtype, device=x.device)


class LpRegularizer(Regularizer):
    """A simple L_p norm based regularizer."""

    #: The dimension along which to compute the vector-based regularization terms.
    dim: Optional[int]

    #: Whether to normalize the regularization term by the dimension of the vectors.
    #: This allows dimensionality-independent weight tuning.
    normalize: bool

    #: The default strategy for optimizing the LP regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
        apply_only_once: bool = False,
        parameters: Optional[Iterable[nn.Parameter]] = None,
    ):
        super().__init__(weight=weight, apply_only_once=apply_only_once, parameters=parameters)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return lp_norm(x=x, p=self.p, dim=self.dim, normalize=self.normalize).mean()


class PowerSumRegularizer(Regularizer):
    """A simple x^p based regularizer.

    Has some nice properties, cf. e.g. https://github.com/pytorch/pytorch/issues/28119.
    """

    #: The default strategy for optimizing the power sum regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
        apply_only_once: bool = False,
        parameters: Optional[Iterable[nn.Parameter]] = None,
    ):
        super().__init__(weight=weight, apply_only_once=apply_only_once, parameters=parameters)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return powersum_norm(x, p=self.p, dim=self.dim, normalize=self.normalize).mean()


class TransHRegularizer(Regularizer):
    """A regularizer for the soft constraints in TransH."""

    #: The default strategy for optimizing the TransH regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        weight: float = 0.05,
        epsilon: float = 1e-5,
        parameters: Optional[Iterable[nn.Parameter]] = None,
    ):
        # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
        # Therefore, apply_only_once is always set to True.
        super().__init__(weight=weight, apply_only_once=True, parameters=parameters)
        self.epsilon = epsilon

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError('TransH regularizer is order-sensitive!')

    def update(self, *tensors: torch.FloatTensor) -> None:  # noqa: D102
        if len(tensors) != 3:
            raise KeyError('Expects exactly three tensors')
        if self.apply_only_once and self.updated:
            return
        entity_embeddings, normal_vector_embeddings, relation_embeddings = tensors
        # Entity soft constraint
        self.regularization_term += torch.sum(functional.relu(torch.norm(entity_embeddings, dim=-1) ** 2 - 1.0))

        # Orthogonality soft constraint
        d_r_n = functional.normalize(relation_embeddings, dim=-1)
        self.regularization_term += torch.sum(
            functional.relu(torch.sum((normal_vector_embeddings * d_r_n) ** 2, dim=-1) - self.epsilon),
        )

        self.updated = True


class CombinedRegularizer(Regularizer):
    """A convex combination of regularizers."""

    # The normalization factor to balance individual regularizers' contribution.
    normalization_factor: torch.FloatTensor

    def __init__(
        self,
        regularizers: Iterable[Regularizer],
        total_weight: float = 1.0,
        apply_only_once: bool = False,
    ):
        super().__init__(weight=total_weight, apply_only_once=apply_only_once)
        self.regularizers = nn.ModuleList(regularizers)
        for r in self.regularizers:
            if isinstance(r, NoRegularizer):
                raise TypeError('Can not combine a no-op regularizer')
        self.register_buffer(name='normalization_factor', tensor=torch.as_tensor(
            sum(r.weight for r in self.regularizers),
        ).reciprocal())

    @property
    def normalize(self):  # noqa: D102
        return any(r.normalize for r in self.regularizers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.normalization_factor * sum(r.weight * r.forward(x) for r in self.regularizers)


regularizer_resolver = Resolver.from_subclasses(
    base=Regularizer,
    default=NoRegularizer,
)
