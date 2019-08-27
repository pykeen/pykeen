# -*- coding: utf-8 -*-

"""Callbacks for click options for building magical KGE model CLIs."""

import sys
from typing import Optional, Type

import click
from torch.optim.optimizer import Optimizer

from .constants import criteria_map, optimizer_map
from ...instance_creation_factories import TriplesFactory
from ...typing import Loss
from ...utils import resolve_device

__all__ = [
    'criterion_callback',
    'optimizer_callback',
    'triples_factory_callback',
    'device_callback',
]


def criterion_callback(_, __, value: str) -> Type[Loss]:
    """Convert the name of the criterion into the appropriate :class:`Loss` class."""
    if value is not None:
        try:
            return criteria_map[value]
        except KeyError:
            click.echo(f'Invalid criterion: {value}')
            sys.exit(0)


def optimizer_callback(_, __, value: str) -> Type[Optimizer]:
    """Convert the name of the optimizer into the appropriate :class:`Optimizer` class."""
    if value is not None:
        return optimizer_map[value]


def triples_factory_callback(_, __, value: Optional[str]) -> Optional[TriplesFactory]:
    """Generate a triples factory using the given path."""
    return value and TriplesFactory(path=value)


def device_callback(_, __, value: Optional[str]):
    """Get the right device."""
    return resolve_device(value)
