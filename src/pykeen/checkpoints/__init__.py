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
and :meth:`pykeen.checkpoints.inspect_schedule`.

Example 1
~~~~~~~~~
Write a checkpoint every 10 steps and keep them all.

.. code-block::

    from pykeen.pipeline import pipeline

    result = pipeline(
        dataset="nations",
        model="mure",
        training_kwargs=dict(
            num_epochs=100,
            callbacks="checkpoint",
            # create one checkpoint every 10 epochs
            callbacks_kwargs=dict(
                schedule="every",
                schedule_kwargs=dict(
                    frequency=10,
                ),
            )
        ),
    )

Example 2
~~~~~~~~~
Write a checkpoint at epoch 1, 7, and 10 and keep them all.

.. code-block::

    from pykeen.pipeline import pipeline

    result = pipeline(
        dataset="nations",
        model="mure",
        training_kwargs=dict(
            num_epochs=10,
            callbacks="checkpoint",
            # create checkpoints at epoch 1, 7, and 10
            callbacks_kwargs=dict(
                schedule="explicit",
                schedule_kwargs=dict(
                    steps=(1, 7, 10)
                ),
            )
        ),
    )

Example 3
~~~~~~~~~
Write a checkpoint avery 5 epochs, but also at epoch 7.

.. code-block::

    from pykeen.pipeline import pipeline

    result = pipeline(
        dataset="nations",
        model="mure",
        training_kwargs=dict(
            num_epochs=10,
            callbacks="checkpoint",
            callbacks_kwargs=dict(
                schedule="union",
                # create checkpoints every 5 epochs, and at epoch 7
                schedule_kwargs=dict(
                    bases=["every", "explicit"],
                    bases_kwargs=[dict(frequency=5), dict(steps=[7])]
                ),
            )
        ),
    )

Example 4
~~~~~~~~~
Write a checkpoint whenever a metric improves (here, just the training loss).

.. code-block::

    from pykeen.checkpoints import MetricSelection
    from pykeen.pipeline import pipeline
    from pykeen.trackers import tracker_resolver

    # create a default result tracker (or use a proper one)
    result_tracker = tracker_resolver.make(None)
    result = pipeline(
        dataset="nations",
        model="mure",
        training_kwargs=dict(
            num_epochs=10,
            callbacks="checkpoint",
            callbacks_kwargs=dict(
                schedule="best",
                schedule_kwargs=dict(
                    result_tracker=result_tracker,
                    # in this example, we just use the training loss
                    metric_selection=MetricSelection(
                        metric="loss",
                        maximize=False,
                    )
                ),
            ),
        ),
        # Important: use the same result tracker instance as in the checkpoint callback
        result_tracker=result_tracker
    )



Example 5
~~~~~~~~~
Write a checkpoint every 10 steps, but keep only the last one and one every 50 steps.

.. code-block::

    from pykeen.pipeline import pipeline

    result = pipeline(
        dataset="nations",
        model="mure",
        training_kwargs=dict(
            num_epochs=100,
            callbacks="checkpoint",
            # create one checkpoint every 10 epochs
            callbacks_kwargs=dict(
                schedule="every",
                schedule_kwargs=dict(
                    frequency=10,
                ),
                keeper="union",
                keeper_kwargs=dict(
                    bases=["modulo", "last"],
                    bases_kwargs=[dict(divisor=50), None],
                )
            )
        ),
    )
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
