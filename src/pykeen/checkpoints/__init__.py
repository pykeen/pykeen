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

To reduce the number of necessary imports, the examples all use dictionaries/strings to
specify components instead of passing classes or actual instances.
You can find more information about resolution in general at :ref:`using_resolvers`.
The resolver for the schedule component is :data:`pykeen.checkpoints.schedule.schedule_resolver`,
and for the keeper component it is :data:`pykeen.checkpoints.keeper_resolver`.

Example 1
~~~~~~~~~
.. literalinclude:: ../examples/checkpoints/ex_01.py

Example 2
~~~~~~~~~
.. literalinclude:: ../examples/checkpoints/ex_02.py

Example 3
~~~~~~~~~
.. literalinclude:: ../examples/checkpoints/ex_03.py

Example 4
~~~~~~~~~
.. literalinclude:: ../examples/checkpoints/ex_04.py

Example 5
~~~~~~~~~
.. literalinclude:: ../examples/checkpoints/ex_05.py
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
