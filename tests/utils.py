# -*- coding: utf-8 -*-

"""Utilities for PyKEEN testing."""

import importlib
import unittest

import torch

__all__ = [
    "rand",
    "needs_packages",
]


def rand(*size: int, generator: torch.Generator, device: torch.device) -> torch.FloatTensor:
    """Wrap generating random numbers with a generator and given device."""
    return torch.rand(*size, generator=generator).to(device=device)


def needs_packages(*names: str) -> unittest.skipIf:
    """Decorate a test such that it only runs if the rqeuired package is available."""
    mods = []
    for name in names:
        try:
            mod = importlib.import_module(name=name)
        except ImportError:
            mod = None
        mods.append(mod)
    return unittest.skipIf(
        condition=any(mod is None for mod in mods), reason=f"Test requires `{names}` to be installed."
    )
