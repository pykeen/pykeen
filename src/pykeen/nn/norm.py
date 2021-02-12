# -*- coding: utf-8 -*-

"""Functional forms of normalization."""

from typing import Optional

import torch
from torch.nn import functional

from ..utils import get_expected_norm

__all__ = [
    'lp_norm',
    'powersum_norm',
    'complex_normalize',
]


def lp_norm(x: torch.FloatTensor, p: float, dim: Optional[int], normalize: bool) -> torch.FloatTensor:
    """Return the $L_p$ norm."""
    value = x.norm(p=p, dim=dim)
    if not normalize:
        return value
    return value / get_expected_norm(p=p, d=x.shape[-1])


def powersum_norm(x: torch.FloatTensor, p: float, dim: Optional[int], normalize: bool) -> torch.FloatTensor:
    """Return the power sum norm."""
    value = x.abs().pow(p).sum(dim=dim)
    if not normalize:
        return value
    dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
    return value / dim


def complex_normalize(x: torch.Tensor) -> torch.Tensor:
    r"""Normalize a vector of complex numbers such that each element is of unit-length.

    The `modulus of complex number <https://en.wikipedia.org/wiki/Absolute_value#Complex_numbers>`_ is given as:

    .. math::

        |a + ib| = \sqrt{a^2 + b^2}

    $l_2$ norm of complex vector $x \in \mathbb{C}^d$:

    .. math::
        \|x\|^2 = \sum_{i=1}^d |x_i|^2
                 = \sum_{i=1}^d \left(\operatorname{Re}(x_i)^2 + \operatorname{Im}(x_i)^2\right)
                 = \left(\sum_{i=1}^d \operatorname{Re}(x_i)^2) + (\sum_{i=1}^d \operatorname{Im}(x_i)^2\right)
                 = \|\operatorname{Re}(x)\|^2 + \|\operatorname{Im}(x)\|^2
                 = \| [\operatorname{Re}(x); \operatorname{Im}(x)] \|^2
    """
    y = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    y = functional.normalize(y, p=2, dim=-1)
    return y.view(*x.shape)
