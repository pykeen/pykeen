# -*- coding: utf-8 -*-

"""Tests for result trackers."""
from pykeen.trackers.base import ConsoleResultTracker
from pykeen.trackers.file import CSVResultTracker, JSONResultTracker
from tests import cases


class CSVResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker


class JSONResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for JSONResultTracker."""

    cls = JSONResultTracker


class ConsoleResultTrackerTests(cases.ResultTrackerTests):
    """Tests for console tracker."""

    cls = ConsoleResultTracker
