# -*- coding: utf-8 -*-

"""Test caching.

This module relies on :func:`pystow.utils.mock_home`, which needs to be used
in a funny way to make sure that things are actually mocked properly.
"""

import unittest

from pystow.utils import mock_home


class TestDatasetCaching(unittest.TestCase):
    """Test caching."""

    def test_it(self):
        """Test dataset caching."""
        digest = "e3b0c44298fc1c149afbf4c8996fb924"
        # The mock_home() is done before importing pykeen so the modules are mocked
        # Don't add more tests in this module.
        with mock_home():
            from pykeen.constants import PYKEEN_DATASETS
            from pykeen.datasets import Nations, _cached_get_dataset

            path = PYKEEN_DATASETS.joinpath(Nations.get_normalized_name(), "cache", digest)
            self.assertFalse(path.is_dir())
            _ = _cached_get_dataset("nations", {})
            self.assertTrue(path.is_dir())
