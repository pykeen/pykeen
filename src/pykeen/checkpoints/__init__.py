"""
This module contains methods for deciding when to write and clear checkpoints.

.. warning ::
    While this module provides a flexible and modular way to describe a desired checkpoint behavior, it currently only
    stores the model's weights (more precisely, its :meth:`torch.nn.Module.state_dict`).
    Thus, it does not yet replace the full training loop checkpointing mechanism described in
    :ref:`regular_checkpoints_how_to`.
"""

from .base import save_model
from .inspection import inspect_schedule
from .keeper import (
    BestCheckpointKeeper,
    CheckpointKeeper,
    ExplicitCheckpointKeeper,
    LastCheckpointKeeper,
    ModuloCheckpointKeeper,
    UnionCheckpointKeeper,
    keeper_resolver,
)
from .schedule import (
    BestCheckpointSchedule,
    CheckpointSchedule,
    EveryCheckpointSchedule,
    ExplicitCheckpointSchedule,
    UnionCheckpointSchedule,
    schedule_resolver,
)
from .utils import MetricSelection

__all__ = [
    "save_model",
    "schedule_resolver",
    "CheckpointSchedule",
    "EveryCheckpointSchedule",
    "ExplicitCheckpointSchedule",
    "BestCheckpointSchedule",
    "UnionCheckpointSchedule",
    "keeper_resolver",
    "CheckpointKeeper",
    "LastCheckpointKeeper",
    "ModuloCheckpointKeeper",
    "ExplicitCheckpointKeeper",
    "BestCheckpointKeeper",
    "UnionCheckpointKeeper",
    "MetricSelection",
    "inspect_schedule",
]
