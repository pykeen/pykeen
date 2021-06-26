# -*- coding: utf-8 -*-

"""Tests for result trackers."""

import unittest

from pykeen.trackers import TensorBoardResultTracker
from pykeen.trackers.base import ConsoleResultTracker
from pykeen.trackers.file import CSVResultTracker, JSONResultTracker
from tests import cases

try:
    import tensorboard
except ImportError:
    tensorboard = None


class CSVResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker


class JSONResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for JSONResultTracker."""

    cls = JSONResultTracker


class ConsoleResultTrackerTests(cases.ResultTrackerTests):
    """Tests for console tracker."""

    cls = ConsoleResultTracker


@unittest.skipIf(tensorboard is None, reason='TensorBoard is not installed')
class TensorboardTrackerTests(cases.ResultTrackerTests):
    """Tests for TensorBoard tracker."""

    cls = TensorBoardResultTracker
