# -*- coding: utf-8 -*-

"""Tests for result trackers."""

from pykeen.trackers import TensorBoardResultTracker
from pykeen.trackers.base import ConsoleResultTracker, MultiResultTracker, PythonResultTracker
from pykeen.trackers.file import CSVResultTracker, JSONResultTracker
from tests import cases

from .utils import needs_packages


class PythonResultTrackerTests(cases.ResultTrackerTests):
    """Tests for Python result tracker."""

    cls = PythonResultTracker


class CSVResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker


class JSONResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for JSONResultTracker."""

    cls = JSONResultTracker


class ConsoleResultTrackerTests(cases.ResultTrackerTests):
    """Tests for console tracker."""

    cls = ConsoleResultTracker


class MultiResultTrackerTests(cases.ResultTrackerTests):
    """Tests for multi tracker."""

    cls = MultiResultTracker
    kwargs = dict(
        trackers=(
            ConsoleResultTracker(),
            ConsoleResultTracker(),
        )
    )


@needs_packages("tensorboard")
class TensorboardTrackerTests(cases.ResultTrackerTests):
    """Tests for TensorBoard tracker."""

    cls = TensorBoardResultTracker
