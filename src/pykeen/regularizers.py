# -*- coding: utf-8 -*-

"""Regularization in PyKEEN."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Collection, Iterable, Mapping, Optional, Sequence, Type, Union

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
    'collect_regularization_terms',
    'get_regularizer_cls',
]

_REGULARIZER_SUFFIX = 'Regularizer'


class Regularizer(nn.Module, ABC):
    """A base class for all regularizers."""

    #: The overall regularization weight
    weight: torch.FloatTensor

    #: The current regularization term (a scalar)
    regularization_term: Union[torch.FloatTensor, float]

    #: Should the regularization only be applied once? This was used for ConvKB and defaults to False.
    apply_only_once: bool

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]]

    #: weights which should be regularized
    tracked_parameters: Optional[Collection[nn.Parameter]] = None

    def __init__(
        self,
        weight: float = 1.0,
        apply_only_once: bool = False,
        parameters: Optional[Sequence[nn.Parameter]] = None,
    ):
        super().__init__()
        self.register_buffer(name='weight', tensor=torch.as_tensor(weight))
        self.apply_only_once = apply_only_once
        self.tracked_parameters = parameters
        self._clear()

    def _clear(self):
        self.regularization_term = 0.
        self.updated = False

    @classmethod
    def get_normalized_name(cls) -> str:
        """Get the normalized name of the regularizer class."""
        return normalize_string(cls.__name__, suffix=_REGULARIZER_SUFFIX)

    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the regularization term for one tensor."""
        raise NotImplementedError

    def update(self, *tensors: torch.FloatTensor) -> bool:
        """Update the regularization term based on passed tensors."""
        if not self.training or not torch.is_grad_enabled() or (self.apply_only_once and self.updated):
            return False
        self.regularization_term = self.regularization_term + sum(self(x) for x in tensors)
        self.updated = True
        return True

    def pop_regularization_term(self) -> torch.FloatTensor:
        """Return the weighted regularization term, and clear it afterwards."""
        if self.tracked_parameters is not None:
            self.update(*self.tracked_parameters)
        term = self.regularization_term
        self._clear()
        return self.weight * term


class NoRegularizer(Regularizer):
    """A regularizer which does not perform any regularization.

    Used to simplify code.
    """
    # TODO: Deprecated

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {}

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        # always return zero
        return x.new_zeros(1)


class LpRegularizer(Regularizer):
    """A simple L_p norm based regularizer."""

    #: The dimension along which to compute the vector-based regularization terms.
    dim: Optional[int]

    #: Whether to normalize the regularization term by the dimension of the vectors.
    #: This allows dimensionality-independent weight tuning.
    normalize: bool

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
        apply_only_once: bool = False,
        parameters: Optional[Sequence[nn.Parameter]] = None,
    ):
        super().__init__(weight=weight, apply_only_once=apply_only_once, parameters=parameters)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        value = x.norm(p=self.p, dim=self.dim).mean()
        if not self.normalize:
            return value
        dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
        if self.p == 1:
            # expected value of |x|_1 = d*E[x_i] for x_i i.i.d.
            return value / dim
        if self.p == 2:
            # expected value of |x|_2 when x_i are normally distributed
            # cf. https://arxiv.org/pdf/1012.0621.pdf chapter 3.1
            return value / dim.sqrt()
        raise NotImplementedError(f'Lp regularization not implemented for p={self.p}')


class PowerSumRegularizer(Regularizer):
    """A simple x^p based regularizer.

    Has some nice properties, cf. e.g. https://github.com/pytorch/pytorch/issues/28119.
    """

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
        apply_only_once: bool = False,
        parameters: Optional[Sequence[nn.Parameter]] = None,
    ):
        super().__init__(weight=weight, apply_only_once=apply_only_once, parameters=parameters)
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

    #: The default strategy for optimizing the regularizer's hyper-parameters
    hpo_default = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale='log'),
    )

    def __init__(
        self,
        entity_embeddings: nn.Parameter,
        normal_vector_embeddings: nn.Parameter,
        relation_embeddings: nn.Parameter,
        weight: float = 0.05,
        epsilon: float = 1e-5,
    ):
        # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
        # Therefore, apply_only_once is always set to True.
        super().__init__(weight=weight, apply_only_once=False, parameters=[])
        self.normal_vector_embeddings = normal_vector_embeddings
        self.relation_embeddings = relation_embeddings
        self.entity_embeddings = entity_embeddings
        self.epsilon = epsilon

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        raise NotImplementedError('TransH regularizer is order-sensitive!')

    def pop_regularization_term(self, *tensors: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        # Entity soft constraint
        self.regularization_term += torch.sum(functional.relu(torch.norm(self.entity_embeddings, dim=-1) ** 2 - 1.0))

        # Orthogonality soft constraint
        d_r_n = functional.normalize(self.relation_embeddings, dim=-1)
        self.regularization_term += torch.sum(
            functional.relu(torch.sum((self.normal_vector_embeddings * d_r_n) ** 2, dim=-1) - self.epsilon),
        )
        return super().pop_regularization_term()


class CombinedRegularizer(Regularizer):
    """A convex combination of regularizers."""

    # The normalization factor to balance individual regularizers' contribution.
    normalization_factor: torch.FloatTensor

    def __init__(
        self,
        regularizers: Iterable[Regularizer],
        total_weight: float = 1.0,
        apply_only_once: bool = False,
        parameters: Optional[Sequence[nn.Parameter]] = None,
    ):
        super().__init__(weight=total_weight, apply_only_once=apply_only_once, parameters=parameters)
        self.regularizers = nn.ModuleList(regularizers)
        for r in self.regularizers:
            if isinstance(r, NoRegularizer):
                raise TypeError('Can not combine a no-op regularizer')
        normalization_factor = torch.as_tensor(sum(r.weight for r in self.regularizers)).reciprocal()
        self.register_buffer(name='normalization_factor', tensor=normalization_factor)

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


def collect_regularization_terms(main_module: nn.Module) -> Union[float, torch.FloatTensor]:
    """Recursively collect regularization terms from attached regularizers, and clear their accumulator."""
    return sum(
        module.pop_regularization_term()
        for module in main_module.modules()
        if isinstance(module, Regularizer)
    )
