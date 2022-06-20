# -*- coding: utf-8 -*-

"""Regularization in PyKEEN."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Mapping, Optional

import torch
from class_resolver import ClassResolver, normalize_string
from torch import nn
from torch.nn import functional

from .utils import lp_norm, powersum_norm

__all__ = [
    # Base Class
    "Regularizer",
    # Child classes
    "LpRegularizer",
    "NoRegularizer",
    "CombinedRegularizer",
    "PowerSumRegularizer",
    "OrthogonalityRegularizer",
    "NormLimitRegularizer",
    # Utils
    "regularizer_resolver",
]

_REGULARIZER_SUFFIX = "Regularizer"


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
        self.register_buffer(name="weight", tensor=torch.as_tensor(weight))
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

        result = self.weight * self.regularization_term
        self.reset()
        return result

    def post_parameter_update(self):
        """
        Reset the regularizer's term.

        .. warning ::
            Typically, you want to use the regularization term exactly once to calculate gradients via
            :meth:`pop_regularization_term`. In this case, there should be no need to manually call this method.
        """
        if self.updated:
            warnings.warn("Resetting regularization term without using it; this may be an error.")
        self.reset()


class NoRegularizer(Regularizer):
    """A regularizer which does not perform any regularization.

    Used to simplify code.
    """

    #: The default strategy for optimizing the no-op regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}

    # docstr-coverage: inherited
    def update(self, *tensors: torch.FloatTensor) -> None:  # noqa: D102
        # no need to compute anything
        pass

    # docstr-coverage: inherited
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
        weight=dict(type=float, low=0.01, high=1.0, scale="log"),
    )

    def __init__(
        self,
        *,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        weight: float = 1.0,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        apply_only_once: bool = False,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.0,
        **kwargs,
    ):
        """
        Initialize the regularizer.

        :param weight:
            The relative weight of the regularization
        :param apply_only_once:
            Should the regularization be applied more than once after reset?
        :param dim:
            the dimension along which to calculate the Lp norm, cf. :func:`lp_norm`
        :param normalize:
            whether to normalize the norm by the dimension, cf. :func:`lp_norm`
        :param p:
            the parameter $p$ of the Lp norm, cf. :func:`lp_norm`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Regularizer.__init__`
        """
        super().__init__(weight=weight, apply_only_once=apply_only_once, **kwargs)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    # docstr-coverage: inherited
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return lp_norm(x=x, p=self.p, dim=self.dim, normalize=self.normalize).mean()


class PowerSumRegularizer(Regularizer):
    """A simple x^p based regularizer.

    Has some nice properties, cf. e.g. https://github.com/pytorch/pytorch/issues/28119.
    """

    #: The default strategy for optimizing the power sum regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale="log"),
    )

    def __init__(
        self,
        *,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        weight: float = 1.0,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        apply_only_once: bool = False,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.0,
        **kwargs,
    ):
        """
        Initialize the regularizer.

        :param weight:
            The relative weight of the regularization
        :param apply_only_once:
            Should the regularization be applied more than once after reset?
        :param dim:
            the dimension along which to calculate the Lp norm, cf. :func:`powersum_norm`
        :param normalize:
            whether to normalize the norm by the dimension, cf. :func:`powersum_norm`
        :param p:
            the parameter $p$ of the Lp norm, cf. :func:`powersum_norm`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Regularizer.__init__`
        """
        super().__init__(weight=weight, apply_only_once=apply_only_once, **kwargs)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    # docstr-coverage: inherited
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return powersum_norm(x, p=self.p, dim=self.dim, normalize=self.normalize).mean()


class NormLimitRegularizer(Regularizer):
    """A regularizer which formulates a soft constraint on a maximum norm."""

    def __init__(
        self,
        *,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        weight: float = 1.0,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        apply_only_once: bool = False,
        # regularizer-specific parameters
        dim: Optional[int] = -1,
        p: float = 2.0,
        power_norm: bool = True,
        max_norm: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the regularizer.

        :param weight:
            The relative weight of the regularization
        :param apply_only_once:
            Should the regularization be applied more than once after reset?
        :param dim:
            the dimension along which to calculate the Lp norm, cf. :func:`powersum_norm`
        :param p:
            the parameter $p$ of the Lp norm, cf. :func:`powersum_norm`
        :param power_norm:
            whether to use the $p$ power of the norm instead
        :param max_norm:
            the maximum norm until which no penalty is added
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Regularizer.__init__`
        """
        super().__init__(weight=weight, apply_only_once=apply_only_once, **kwargs)
        self.dim = dim
        self.p = p
        self.max_norm = max_norm
        self.power_norm = power_norm

    # docstr-coverage: inherited
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        if self.power_norm:
            norm = powersum_norm(x, p=self.p, dim=self.dim, normalize=False)
        else:
            norm = lp_norm(x=x, p=self.p, dim=self.dim, normalize=False)
        return (norm - self.max_norm).relu().sum()


class OrthogonalityRegularizer(Regularizer):
    """A regularizer for the soft orthogonality constraints from [wang2014]_."""

    #: The default strategy for optimizing the TransH regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale="log"),
    )

    def __init__(
        self,
        *,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        weight: float = 1.0,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        apply_only_once: bool = True,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        """
        Initialize the regularizer.

        :param weight:
            The relative weight of the regularization
        :param apply_only_once:
            Should the regularization be applied more than once after reset?
        :param epsilon:
            a small value used to check for approximate orthogonality
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Regularizer.__init__`
        """
        super().__init__(weight=weight, **kwargs, apply_only_once=apply_only_once)
        self.epsilon = epsilon

    # docstr-coverage: inherited
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError(f"{self.__class__.__name__} regularizer is order-sensitive!")

    # docstr-coverage: inherited
    def update(self, *tensors: torch.FloatTensor) -> None:  # noqa: D102
        if len(tensors) != 2:
            raise ValueError("Expects exactly two tensors")
        if self.apply_only_once and self.updated:
            return
        # orthogonality soft constraint: cosine similarity at most epsilon
        self.regularization_term = self.regularization_term + (
            functional.cosine_similarity(*tensors, dim=-1).pow(2).subtract(self.epsilon).relu().sum()
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
        **kwargs,
    ):
        """
        Initialize the regularizer.

        :param regularizers:
            the base regularizers
        :param total_weight:
            the total regularization weight distributed to the base regularizers according to their individual weights
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Regularizer.__init__`
        :raises TypeError:
            if any of the regularizers are a no-op regularizer
        """
        super().__init__(weight=total_weight, **kwargs)
        self.regularizers = nn.ModuleList(regularizers)
        for r in self.regularizers:
            if isinstance(r, NoRegularizer):
                raise TypeError("Can not combine a no-op regularizer")
        self.register_buffer(
            name="normalization_factor",
            tensor=torch.as_tensor(
                sum(r.weight for r in self.regularizers),
            ).reciprocal(),
        )

    # docstr-coverage: inherited
    @property
    def normalize(self):  # noqa: D102
        return any(r.normalize for r in self.regularizers)

    # docstr-coverage: inherited
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.normalization_factor * sum(r.weight * r.forward(x) for r in self.regularizers)


regularizer_resolver: ClassResolver[Regularizer] = ClassResolver.from_subclasses(
    base=Regularizer,
    default=NoRegularizer,
)
