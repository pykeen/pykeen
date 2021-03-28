# -*- coding: utf-8 -*-

"""Utilities for PyKEEN testing."""

import torch

__all__ = [
    'rand',
]


def rand(*size: int, generator: torch.Generator, device: torch.device) -> torch.FloatTensor:
    """Wrap generating random numbers with a generator and given device."""
    return torch.rand(*size, generator=generator).to(device=device)
