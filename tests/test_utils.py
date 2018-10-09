# -*- coding: utf-8 -*-

"""Tests for `utils`."""

import unittest

from pykeen.constants import VERSION


class SimpleTest(unittest.TestCase):
    """Simple sanity tests."""

    def test_version(self):
        """Check the version string can be loaded."""
        self.assertIsInstance(VERSION, str)
