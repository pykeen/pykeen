# -*- coding: utf-8 -*-

"""Functional forms of normalization."""

from typing import Optional

import torch

from ..utils import get_expected_norm

__all__ = [
    'lp_norm',
    'powersum_norm',
]


def lp_norm(x: torch.FloatTensor, p: float, dim: Optional[int], normalize: bool) -> torch.FloatTensor:
    """Return the $L_p$ norm."""
    value = x.norm(p=p, dim=dim).mean()
    if not normalize:
        return value
    return value / get_expected_norm(p=p, d=x.shape[-1])


def powersum_norm(x: torch.FloatTensor, p: float, dim: Optional[int], normalize: bool) -> torch.FloatTensor:
    """Return the power sum norm."""
    value = x.abs().pow(p).sum(dim=dim).mean()
    if not normalize:
        return value
    dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
    return value / dim
