"""Tests for checkpointing."""

from collections.abc import Iterator, MutableMapping
from typing import Any, ClassVar

import torch
import unittest_templates

from pykeen.checkpoints import keeper, schedule
from pykeen.checkpoints.utils import MetricSelection
from pykeen.trackers.base import PythonResultTracker
from tests.cases import CheckpointKeeperTests, CheckpointScheduleTests


class CheckpointScheduleMetaTestCase(unittest_templates.MetaTestCase[schedule.CheckpointSchedule]):
    """Meta test case for checkpoint schedules."""

    base_cls: ClassVar = schedule.CheckpointSchedule
    base_test: ClassVar = CheckpointScheduleTests


class EveryCheckpointScheduleTests(CheckpointScheduleTests):
    """Test for every."""

    cls = schedule.EveryCheckpointSchedule


class ExplicitCheckpointScheduleTests(CheckpointScheduleTests):
    """Test for explicit."""

    cls = schedule.ExplicitCheckpointSchedule
    kwargs = dict(steps=(4, 6))


class BestCheckpointScheduleTests(CheckpointScheduleTests):
    """Test for best."""

    cls = schedule.BestCheckpointSchedule
    kwargs = dict(metric_selection=MetricSelection(metric="loss", prefix="validation"))

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["result_tracker"] = self.result_tracker = PythonResultTracker()
        return kwargs

    def iter_steps(self) -> Iterator[int]:
        for step in super().iter_steps():
            loss = torch.rand(1, generator=self.generator)
            self.result_tracker.log_metrics(metrics=dict(loss=loss), step=step, prefix="validation")
            yield step


class UnionCheckpointScheduleTests(CheckpointScheduleTests):
    """Test for union."""

    cls = schedule.UnionCheckpointSchedule
    kwargs = dict(bases=["every", "explicit"], bases_kwargs=[None, dict(steps=(3,))])


class CheckpointKeeperMetaTestCase(unittest_templates.MetaTestCase[keeper.CheckpointKeeper]):
    """Meta test case for checkpoint keepers."""

    base_cls: ClassVar = keeper.CheckpointKeeper
    base_test: ClassVar = CheckpointKeeperTests


class ExplicitCheckpointKeeperTests(CheckpointKeeperTests):
    """Tests for explicit."""

    cls = keeper.ExplicitCheckpointKeeper
    kwargs = dict(keep=(3, 6))


class LastCheckpointKeeperTests(CheckpointKeeperTests):
    """Tests for last."""

    cls = keeper.LastCheckpointKeeper


class BestCheckpointKeeperTests(CheckpointKeeperTests):
    """Tests for best."""

    cls = keeper.BestCheckpointKeeper
    kwargs = dict(metric_selection=MetricSelection(metric="loss", prefix="validation"))

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["result_tracker"] = self.result_tracker = PythonResultTracker()
        return kwargs

    def iter_steps(self) -> Iterator[list[int]]:
        for steps in super().iter_steps():
            losses = torch.rand(len(steps), generator=self.generator)
            for step, loss in zip(steps, losses.tolist()):
                self.result_tracker.log_metrics(metrics=dict(loss=loss), step=step, prefix="validation")
            yield steps


class ModuloCheckpointKeeperTests(CheckpointKeeperTests):
    """Tests for modulo."""

    cls = keeper.ModuloCheckpointKeeper


class UnionCheckpointKeeperTests(CheckpointKeeperTests):
    """Tests for union."""

    cls = keeper.UnionCheckpointKeeper
    kwargs = dict(
        bases=["last", "explicit"],
        bases_kwargs=[
            dict(keep=1),
            dict(keep=(3, 7)),
        ],
    )
