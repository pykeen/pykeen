# -*- coding: utf-8 -*-

"""Tests for result trackers."""

from pykeen.trackers.file import CSVResultTracker
from tests import cases


class CSVResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker
