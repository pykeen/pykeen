"""
This module contains methods for deciding when to write and clear checkpoints.

.. warning ::
    While this module provides a flexible and modular way to describe a desired checkpoint behavior, it currently only
    stores the model's weights (more precisely, its :meth:`torch.nn.Module.state_dict`).
    Thus, it does not yet replace the full training loop checkpointing mechanism described in
    :ref:`regular_checkpoints_how_to`.

It consists of two main components: checkpoint *schedules* decide whether to write a checkpoint at a given epoch.
If we have multiple checkpoints, we can use multiple *keep strategies* to decide which checkpoints to keep and
which to discard. For both, we provide a set of basic rules, as well as a way to combine them via union.
Those should be sufficient to easily model most of the desired checkpointing behaviours.

Examples
========

Below you can find a few examples of how to use them inside the training pipeline.
If you want to check before an actual training how (static) checkpoint schedules behave,
you can take a look at :meth:`pykeen.checkpoints.final_checkpoints`
and :meth:`pykeen.checkpoints.simulate_checkpoints`.

Example 1
~~~~~~~~~
Write a checkpoint every 10 steps and keep them all.

.. literalinclude:: ../examples/checkpoints_01.py

Example 2
~~~~~~~~~
Write a checkpoint at epoch 1, 7, and 10 and keep them all.

.. literalinclude:: ../examples/checkpoints_02.py

Example 3
~~~~~~~~~
Write a checkpoint avery 5 epochs, but also at epoch 7.

.. literalinclude:: ../examples/checkpoints_03.py

Example 4
~~~~~~~~~
Write a checkpoint whenever a metric improves (here, just the training loss).

.. literalinclude:: ../examples/checkpoints_04.py


Example 5
~~~~~~~~~
Write a checkpoint every 10 steps, but keep only the last one and one every 50 steps.

.. literalinclude:: ../examples/checkpoints_05.py
"""

from .base import save_model
from .inspection import final_checkpoints, simulate_checkpoints
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
    "simulate_checkpoints",
    "final_checkpoints",
]
