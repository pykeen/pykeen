"""Tools for investigating schedules outside of a training session."""

from class_resolver import HintOrType, OptionalKwargs

from .keeper import CheckpointKeeper, keeper_resolver
from .schedule import CheckpointSchedule, schedule_resolver


def simulate_checkpoints(
    num_epochs: int = 100,
    schedule: HintOrType[CheckpointSchedule] = None,
    schedule_kwargs: OptionalKwargs = None,
    keeper: HintOrType[CheckpointKeeper] = None,
    keeper_kwargs: OptionalKwargs = None,
) -> None:
    """
    Simulate a checkpoint schedule and print information about checkpointing.

    .. warning::
        You cannot easily simulate schedules which depend on training dynamics, e.g., :class:`BestCheckpointSchedule`.

    :param num_epochs:
        the number of epochs
    :param schedule:
        a checkpoint schedule instance or selection, cf. :const:`pykeen.checkpoints.scheduler_resolver`
    :param schedule_kwargs:
        additional keyword-based parameters when the schedule needs to instantiated first from a selection,
        cf. :const:`pykeen.checkpoints.scheduler_resolver`
    :param keeper:
        a checkpoint retention policy instance or selection, cf. :const:`pykeen.checkpoints.keeper_resolver`
        `None` corresponds to keeping everything that was checkpointed.
    :param keeper_kwargs:
        additional keyword-based parameters when the retention policy needs to instantiated first from a selection,
        cf. :const:`pykeen.checkpoints.keeper_resolver`
    """
    schedule_instance = schedule_resolver.make(schedule, schedule_kwargs)
    keeper_instance = keeper_resolver.make_safe(keeper, keeper_kwargs)
    checkpoints: list[int] = []
    for epoch in range(1, num_epochs + 1):
        if schedule_instance(epoch):
            print(f"Write checkpoint at {epoch=}")  # noqa: T201
        checkpoints.append(epoch)
        if not keeper_instance:
            continue
        to_keep = keeper_instance(steps=checkpoints)
        for checkpoint in checkpoints:
            if checkpoint not in to_keep:
                print(f"Delete checkpoint at {epoch=}")  # noqa: T201
        checkpoints = sorted(to_keep)


def final_checkpoints(
    num_epochs: int = 100,
    schedule: HintOrType[CheckpointSchedule] = None,
    schedule_kwargs: OptionalKwargs = None,
    keeper: HintOrType[CheckpointKeeper] = None,
    keeper_kwargs: OptionalKwargs = None,
) -> list[int]:
    """
    Simulate a checkpoint schedule and return the set of epochs for which a checkpoint remains.

    >>> final_checkpoints(50)
    [10, 20, 30, 40, 50]
    >>> final_checkpoints(50, schedule="explicit", schedule_kwargs=dict(steps=[30, 35]))
    [30, 35]
    >>> final_checkpoints(
    ...     50,
    ...     schedule="union",
    ...     schedule_kwargs=dict(
    ...         bases=["every", "explicit"],
    ...         bases_kwargs=[dict(frequency=15), dict(steps=[7,])],
    ...     ),
    ... )
    [7, 15, 30, 45]
    >>> final_checkpoints(50, keeper="last", keeper_kwargs=dict(keep=2))
    [40, 50]
    >>> final_checkpoints(50, keeper="modulo", keeper_kwargs=dict(modulo=20))
    [20, 40]
    >>> final_checkpoints(50, keeper="explicit", keeper_kwargs=dict(keep=[15]))
    []
    >>> final_checkpoints(
    ...     50,
    ...     keeper="union",
    ...     keeper_kwargs=dict(
    ...         bases=["last", "modulo"],
    ...         bases_kwargs=[None, dict(divisor=20)],
    ...     ),
    ... )
    [20, 40, 50]

    .. warning::
        You cannot easily inspect schedules which depend on training dynamics, e.g., :class:`BestCheckpointSchedule`.

    :param num_epochs:
        the number of epochs
    :param schedule:
        a checkpoint schedule instance or selection, cf. :const:`pykeen.checkpoints.scheduler_resolver`
    :param schedule_kwargs:
        additional keyword-based parameters when the schedule needs to instantiated first from a selection,
        cf. :const:`pykeen.checkpoints.scheduler_resolver`
    :param keeper:
        a checkpoint retention policy instance or selection, cf. :const:`pykeen.checkpoints.keeper_resolver`
        `None` corresponds to keeping everything that was checkpointed.
    :param keeper_kwargs:
        additional keyword-based parameters when the retention policy needs to instantiated first from a selection,
        cf. :const:`pykeen.checkpoints.keeper_resolver`

    :return:
        a sorted list of epochs at which a checkpoint remains after clean-up.
    """
    schedule_instance = schedule_resolver.make(schedule, schedule_kwargs)
    keeper_instance = keeper_resolver.make_safe(keeper, keeper_kwargs)
    epochs = range(1, num_epochs + 1)

    # determine when checkpoints are written
    checkpoint_epochs = filter(schedule_instance, epochs)
    if not keeper_instance:
        return list(checkpoint_epochs)

    # simulate cleanup
    remaining: list[int] = []
    for epoch in checkpoint_epochs:
        remaining = sorted(keeper_instance(steps=remaining + [epoch]))
    return list(remaining)
