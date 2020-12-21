# -*- coding: utf-8 -*-

"""Regularization in PyKEEN."""
import functools
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Collection, Iterable, Mapping, Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional

from .utils import get_cls, normalize_string

__all__ = [
    'Regularizer',
    'LpRegularizer',
    'NoRegularizer',
    'CombinedRegularizer',
    'PowerSumRegularizer',
    'TransHRegularizer',
    'get_regularizer_cls',
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

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]

    def __init__(
        self,
        device: torch.device,
        weight: float = 1.0,
        apply_only_once: bool = False,
    ):
        super().__init__()
        self.device = device
        self.register_buffer(name='weight', tensor=torch.as_tensor(weight, device=self.device))
        self.apply_only_once = apply_only_once
        self.reset()

    def to(self, *args, **kwargs) -> 'Regularizer':  # noqa: D102
        super().to(*args, **kwargs)
        self.device = torch._C._nn._parse_to(*args, **kwargs)[0]
        self.reset()
        return self

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the regularizer class."""
        return normalize_string(cls.__name__, suffix=_REGULARIZER_SUFFIX)

    def reset(self) -> None:
        """Reset the regularization term to zero."""
        self.regularization_term = torch.zeros(1, dtype=torch.float, device=self.device)
        self.updated = False

    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the regularization term for one tensor."""
        raise NotImplementedError

    def update(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularization term based on passed tensors."""
        if self.apply_only_once and self.updated:
            return
        self.regularization_term = self.regularization_term + sum(self.forward(x=x) for x in tensors)
        self.updated = True

    @property
    def term(self) -> torch.FloatTensor:
        """Return the weighted regularization term."""
        return self.regularization_term * self.weight


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


@functools.lru_cache(maxsize=1)
def _get_expected_norm(
    p: Union[int, float, str],
    d: int,
) -> float:
    r"""
    Compute the expected value of the L_p norm.

    .. math ::
        E[\|x\|_p] = d^{1/p} E[|x_1|^p]^{1/p}

    under the assumption that :math:`x_i \sim N(0, 1)`, i.e.

    .. math ::
        E[|x_1|^p] = 2^{p/2} \cdot \Gamma(\frac{p+1}{2} \cdot \pi^{-1/2}

    :param p:
        The parameter p of the norm.
    :param d:
        The dimension of the vector.

    :return:
        The expected value.

    .. seealso ::
        https://math.stackexchange.com/questions/229033/lp-norm-of-multivariate-standard-normal-random-variable
        https://www.wolframalpha.com/input/?i=expected+value+of+%7Cx%7C%5Ep
    """
    if isinstance(p, str) or not math.isfinite(p):
        # TODO: Use https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution for p = +inf
        raise NotImplementedError(f"{p} norm not implemented")
    exp_abs_norm_p = math.pow(2, p / 2) * math.gamma((p + 1) / 2) / math.sqrt(math.pi)
    return math.pow(exp_abs_norm_p * d, 1 / p)


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
        device: torch.device,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
        apply_only_once: bool = False,
    ):
        super().__init__(device=device, weight=weight, apply_only_once=apply_only_once)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        value = x.norm(p=self.p, dim=self.dim).mean()
        if not self.normalize:
            return value
        return value / _get_expected_norm(p=self.p, d=x.shape[-1])


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
        device: torch.device,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
        apply_only_once: bool = False,
    ):
        super().__init__(device=device, weight=weight, apply_only_once=apply_only_once)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        value = x.abs().pow(self.p).sum(dim=self.dim).mean()
        if not self.normalize:
            return value
        dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
        return value / dim


class TransHRegularizer(Regularizer):
    """A regularizer for the soft constraints in TransH."""

    #: The default strategy for optimizing the TransH regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        device: torch.device,
        weight: float = 0.05,
        epsilon: float = 1e-5,
    ):
        # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
        # Therefore, apply_only_once is always set to True.
        super().__init__(device=device, weight=weight, apply_only_once=True)
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
        device: torch.device,
        total_weight: float = 1.0,
        apply_only_once: bool = False,
    ):
        super().__init__(weight=total_weight, device=device, apply_only_once=apply_only_once)
        self.regularizers = nn.ModuleList(regularizers)
        for r in self.regularizers:
            if isinstance(r, NoRegularizer):
                raise TypeError('Can not combine a no-op regularizer')
        self.register_buffer(name='normalization_factor', tensor=torch.as_tensor(
            sum(r.weight for r in self.regularizers), device=device,
        ).reciprocal())

    @property
    def normalize(self):  # noqa: D102
        return any(r.normalize for r in self.regularizers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.normalization_factor * sum(r.weight * r.forward(x) for r in self.regularizers)


_REGULARIZERS: Collection[Type[Regularizer]] = {
    NoRegularizer,  # type: ignore
    LpRegularizer,
    PowerSumRegularizer,
    CombinedRegularizer,
    TransHRegularizer,
}

#: A mapping of regularizers' names to their implementations
regularizers: Mapping[str, Type[Regularizer]] = {
    cls.get_normalized_name(): cls
    for cls in _REGULARIZERS
}


def get_regularizer_cls(query: Union[None, str, Type[Regularizer]]) -> Type[Regularizer]:
    """Get the regularizer class."""
    return get_cls(
        query,
        base=Regularizer,  # type: ignore
        lookup_dict=regularizers,
        default=NoRegularizer,
        suffix=_REGULARIZER_SUFFIX,
    )
