"""Test caching."""

import pathlib
import tempfile
import unittest

from pykeen.datasets import Nations
from pykeen.datasets.base import Dataset


class TestDatasetCaching(unittest.TestCase):
    """Test caching."""

    def test_serialization(self):
        """Test dataset serialization."""
        dataset = Nations()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            dataset.to_directory_binary(path=path)
            dataset2 = Dataset.from_directory_binary(path=path)
            assert dataset == dataset2
