"""Utilities for PyKEEN testing."""

import importlib
import unittest
from collections.abc import Callable
from typing import TypeVar

import torch

__all__ = [
    "needs_packages",
    "rand",
]

_F = TypeVar("_F", bound=Callable)


def rand(*size: int, generator: torch.Generator, device: torch.device) -> torch.FloatTensor:
    """Wrap generating random numbers with a generator and given device."""
    return torch.rand(*size, generator=generator).to(device=device)


def is_installed(name: str) -> bool:
    """Return whether a package is installed.

    :param name: the package's name

    :returns: whether the package can be imported
    """
    try:
        importlib.import_module(name=name)
    except ImportError:
        return False
    return True


def needs_packages(*names: str) -> Callable[[_F], _F]:
    """Decorate a test such that it only runs if the rqeuired package is available."""
    missing = {name for name in names if not is_installed(name=name)}
    return unittest.skipIf(condition=missing, reason=f"Missing required packages: {sorted(missing)}.")
