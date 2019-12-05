# -*- coding: utf-8 -*-

"""Regularization.

========  ==============================================
Name      Reference
========  ==============================================
combined  :class:`poem.regularizers.CombinedRegularizer`
lp        :class:`poem.regularizers.LpRegularizer`
no        :class:`poem.regularizers.NoRegularizer`
powersum  :class:`poem.regularizers.PowerSumRegularizer`
========  ==============================================

.. note:: This table can be re-generated with ``poem ls regularizers -f rst``
"""

from abc import abstractmethod
from typing import Collection, Iterable, Mapping, Optional, Type, Union

import torch
from torch import nn

from .utils import get_cls, normalize_string

__all__ = [
    'Regularizer',
    'LpRegularizer',
    'NoRegularizer',
    'CombinedRegularizer',
    'PowerSumRegularizer',
    'regularizers',
    'get_regularizer_cls',
]


class Regularizer(nn.Module):
    """A base class for all regularizers."""

    #: The overall regularization weight
    weight: torch.FloatTensor

    #: The current regularization term (a scalar)
    regularization_term: torch.FloatTensor

    def __init__(
        self,
        device: torch.device,
        weight: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.regularization_term = torch.zeros(1, dtype=torch.float, device=self.device)
        self.weight = torch.as_tensor(weight, device=self.device)

    def reset(self) -> None:
        """Reset the regularization term to zero."""
        self.regularization_term = torch.zeros(1, dtype=torch.float, device=self.device)

    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the regularization term for one tensor."""
        raise NotImplementedError

    def update(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularization term based on passed tensors."""
        self.regularization_term = self.regularization_term + sum(self.forward(x=x) for x in tensors)

    @property
    def term(self) -> torch.FloatTensor:
        """Return the weighted regularization term."""
        return self.regularization_term * self.weight


class NoRegularizer(Regularizer):
    """A regularizer which does not perform any regularization.

    Used to simplify code.
    """

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

    def __init__(
        self,
        device: torch.device,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
    ):
        super().__init__(device=device, weight=weight)
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

    def __init__(
        self,
        device: torch.device,
        weight: float = 1.0,
        dim: Optional[int] = -1,
        normalize: bool = False,
        p: float = 2.,
    ):
        super().__init__(device=device, weight=weight)
        self.dim = dim
        self.normalize = normalize
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        value = x.abs().pow(self.p).sum(dim=self.dim).mean()
        if not self.normalize:
            return value
        dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
        return value / dim


class CombinedRegularizer(Regularizer):
    """A convex combination of regularizers."""

    def __init__(
        self,
        regularizers: Iterable[Regularizer],
        device: torch.device,
        total_weight: float = 1.0,
    ):
        super().__init__(weight=total_weight, device=device)
        self.regularizers = list(regularizers)
        for r in self.regularizers:
            if isinstance(r, NoRegularizer):
                raise TypeError('Can not combine a no-op regularizer')
        self.normalization_factor = torch.reciprocal(torch.as_tensor(sum(r.weight for r in self.regularizers)))

    @property
    def normalize(self):  # noqa: D102
        return any(r.normalize for r in self.regularizers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.normalization_factor * sum(r.weight * r.forward(x) for r in self.regularizers)


_REGULARIZER_SUFFIX = 'Regularizer'
_REGULARIZERS: Collection[Type[Regularizer]] = {
    NoRegularizer,
    LpRegularizer,
    PowerSumRegularizer,
    CombinedRegularizer,
}

regularizers: Mapping[str, Type[Regularizer]] = {
    normalize_string(_regularizer.__name__, suffix=_REGULARIZER_SUFFIX): _regularizer
    for _regularizer in _REGULARIZERS
}


def get_regularizer_cls(query: Union[None, str, Type[Regularizer]]) -> Type[Regularizer]:
    """Get the regularizer class."""
    return get_cls(
        query,
        base=Regularizer,
        lookup_dict=regularizers,
        default=NoRegularizer,
        suffix=_REGULARIZER_SUFFIX,
    )
