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


def is_installed(name: str) -> bool:
    """
    Return whether a package is installed.

    :param name:
        the package's name

    :return:
        whether the package can be imported
    """
    try:
        importlib.import_module(name=name)
    except ImportError:
        return False
    return True


def needs_packages(*names: str) -> unittest.skipIf:
    """Decorate a test such that it only runs if the rqeuired package is available."""
    missing = {name for name in names if not is_installed(name=name)}
    return unittest.skipIf(condition=missing, reason=f"Missing required packages: {sorted(missing)}.")
