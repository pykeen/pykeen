# -*- coding: utf-8 -*-

"""Test caching."""

import pathlib
import shutil
import tempfile
import unittest
from timeit import default_timer

from pykeen.constants import PYKEEN_DATASETS
from pykeen.datasets import Nations
from pykeen.datasets.base import Dataset
from pykeen.datasets.utils import _cached_get_dataset, _digest_kwargs


def _time_cached_get_dataset(name: str) -> float:
    start = default_timer()
    _cached_get_dataset(name, {})
    return default_timer() - start


class TestDatasetCaching(unittest.TestCase):
    """Test caching."""

    def test_caching(self):
        """Test dataset caching."""
        digest = _digest_kwargs(dict())
        directory = PYKEEN_DATASETS.joinpath(Nations().get_normalized_name(), "cache", digest)
        # clear
        if directory.exists():
            shutil.rmtree(directory)
        t1 = _time_cached_get_dataset("nations")
        t2 = _time_cached_get_dataset("nations")
        assert t2 < t1 + 1.0e-04

    def test_serialization(self):
        """Test dataset serialization."""
        dataset = Nations()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            dataset.to_directory_binary(path=path)
            dataset2 = Dataset.from_directory_binary(path=path)
            assert dataset == dataset2
