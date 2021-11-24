# -*- coding: utf-8 -*-

"""Test caching."""

import unittest

from pykeen.constants import PYKEEN_DATASETS
from pykeen.datasets import Nations, _cached_get_dataset


class TestDatasetCaching(unittest.TestCase):
    """Test caching."""

    def test_caching(self):
        """Test dataset caching."""
        digest = "e3b0c44298fc1c149afbf4c8996fb924"
        directory = PYKEEN_DATASETS.joinpath(Nations.get_normalized_name(), "cache", digest)
        if directory.is_dir():
            for name in ("training", "testing", "validation"):
                path = directory.joinpath(name).with_suffix(".pt")
                if path.is_file():
                    path.unlink()
            directory.rmdir()

        self.assertFalse(directory.is_dir())
        _ = _cached_get_dataset("nations", {})
        self.assertTrue(directory.is_dir())
