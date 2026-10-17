"""Utilities for PyKEEN testing."""

import importlib
import unittest
from collections.abc import Callable
from typing import TypeVar

import torch

from pykeen.models import Model, TargetScoringBatch, TripleScoringBatch
from pykeen.typing import FloatTensor, LongTensor, Target

__all__ = [
    "forward_hrt",
    "forward_pairs",
    "needs_packages",
    "rand",
]

_F = TypeVar("_F", bound=Callable)


def rand(*size: int, generator: torch.Generator, device: torch.device) -> torch.FloatTensor:
    """Wrap generating random numbers with a generator and given device."""
    return torch.rand(*size, generator=generator).to(device=device)


def forward_hrt(model: Model, hrt_batch: LongTensor, **kwargs) -> FloatTensor:
    """Score triples with *internal* relation IDs via the model's forward pass.

    :param model: the model
    :param hrt_batch: shape: (batch_size, 3)
        the triples
    :param kwargs: additional keyword-based parameters passed to :meth:`pykeen.models.Model.forward`

    :returns: shape: (batch_size,)
        the scores
    """
    return model(TripleScoringBatch.from_batch(hrt_batch), **kwargs)


def forward_pairs(
    model: Model, pairs: LongTensor, target: Target, ids: LongTensor | None = None, **kwargs
) -> FloatTensor:
    """Score the target against all or the given candidates via the model's forward pass, with internal relation IDs.

    :param model: the model
    :param pairs: shape: (batch_size, 2)
        the non-target index pairs, in the order of the columns
    :param target: the position to score against many candidates
    :param ids: shape: (num,) | (batch_size, num)
        the candidates, or None to score against all of them
    :param kwargs: additional keyword-based parameters passed to :meth:`pykeen.models.Model.forward`

    :returns: shape: (batch_size, num)
        the scores
    """
    return model(TargetScoringBatch.from_pairs(pairs, target=target, target_ids=ids), **kwargs)


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
