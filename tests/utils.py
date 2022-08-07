# -*- coding: utf-8 -*-

"""Utilities for PyKEEN testing."""

import importlib
import unittest

import torch

__all__ = [
    "rand",
    "needs_package",
]


def rand(*size: int, generator: torch.Generator, device: torch.device) -> torch.FloatTensor:
    """Wrap generating random numbers with a generator and given device."""
    return torch.rand(*size, generator=generator).to(device=device)


def needs_package(name: str):
    """Decorate a test such that it only runs if the rqeuired package is available."""
    try:
        mod = importlib.import_module(name=name)
    except ImportError:
        mod = None
    return unittest.skipIf(condition=mod is None, reason=f"Test requires `{name}` to be installed.")
