# coding=utf-8

"""Regularization."""

from abc import abstractmethod
from typing import Iterable, Optional, Union

import torch
from torch import nn

from poem.utils import resolve_device

__all__ = [
    'Regularizer',
    'LpRegularizer',
    'NoRegularizer',
    'CombinedRegularizer',
    'PowerSumRegularizer',
]


class Regularizer(nn.Module):
    """A base class for all regularizers."""

    #: The overall regularization weight
    weight: torch.FloatTensor

    #: The current regularization term (a scalar)
    regularization_term: torch.FloatTensor

    #: Whether to normalize the regularization term of individual tensors by the number of elements
    #: This allows dimensionality-independent weight tuning.
    normalize: bool

    def __init__(
        self,
        weight: float = 1.0,
        normalize: bool = False,
        preferred_device: Optional[str] = None,

    ):
        super().__init__()
        # Initialize the device
        self._set_device(preferred_device)
        self.regularization_term = torch.zeros(1, dtype=torch.float, device=self.device)
        self.weight = torch.as_tensor(weight, device=self.device)
        self.normalize = normalize

    def reset(self) -> None:
        """Reset the regularization term to zero."""
        self.regularization_term = torch.zeros(1, dtype=torch.float, device=self.device)

    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the regularization term for one tensor."""
        raise NotImplementedError

    def update(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularization term based on passed tensors."""
        for x in tensors:
            # compute regularization term for a specific tensor
            one_tensor_term = self.forward(x=x)

            # Normalize by the number of elements in the tensors for dimensionality-independent weight tuning.
            if self.normalize:
                one_tensor_term = one_tensor_term / x.numel()

            # Update regularization term
            self.regularization_term += one_tensor_term

    @property
    def term(self) -> torch.FloatTensor:
        """Return the weighted regularization term."""
        return self.regularization_term * self.weight

    def _set_device(self, device: Union[None, str, torch.device] = None) -> None:
        """Set the Torch device to use."""
        self.device = resolve_device(device=device)

    def to_device_(self) -> 'BaseModule':
        """Transfer model to device."""
        self.to(self.device)
        torch.cuda.empty_cache()
        return self


class NoRegularizer(Regularizer):
    """A regularizer which does not perform any regularization.

    Used to simplify code.
    """

    def __init__(self, preferred_device: Optional[str] = None):
        """Initialize NoRegularizer."""
        super().__init__(preferred_device=preferred_device)

    def update(self, *tensors: torch.FloatTensor) -> None:  # noqa: D102
        # no need to compute anything
        pass

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        # always return zero
        return torch.zeros(1, dtype=x.dtype, device=x.device)


class LpRegularizer(Regularizer):
    """A simple L_p norm based regularizer."""

    def __init__(
        self,
        weight: float = 1.0,
        p: float = 2., normalize: bool = False,
        preferred_device: Optional[str] = None
    ):
        super().__init__(weight=weight, normalize=normalize, preferred_device=preferred_device)
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return torch.norm(x, p=self.p)


class PowerSumRegularizer(Regularizer):
    """A simple x^p based regularizer.

    Has some nice properties, cf. e.g. https://github.com/pytorch/pytorch/issues/28119.
    """

    def __init__(
        self,
        weight: float = 1.0,
        p: float = 2.,
        normalize: bool = False,
        preferred_device: Optional[str] = None):
        super().__init__(weight=weight, normalize=normalize, preferred_device=preferred_device)
        self.p = p

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return torch.sum(torch.abs(x) ** self.p)


class CombinedRegularizer(Regularizer):
    """A linear combination of regularizers."""

    def __init__(
        self,
        regularizers: Iterable[Regularizer],
        total_weight: float = 1.0,
        preferred_device: Optional[str] = None
    ):
        super().__init__(weight=total_weight, preferred_device=preferred_device)
        self.regularizers = list(regularizers)
        self.normalization_factor = torch.tensor(1. / sum(r.weight for r in regularizers))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return self.normalization_factor * sum(r.weight * r.forward(x) for r in self.regularizers)
