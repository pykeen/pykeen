# -*- coding: utf-8 -*-

"""Test caching."""

import tempfile
import pathlib
import unittest

import numpy.testing

from pykeen.constants import PYKEEN_DATASETS
from pykeen.datasets import Nations, _cached_get_dataset, _digest_kwargs
from pykeen.datasets.base import Dataset


class TestDatasetCaching(unittest.TestCase):
    """Test caching."""

    def test_caching(self):
        """Test dataset caching."""
        digest = _digest_kwargs(dict())
        directory = PYKEEN_DATASETS.joinpath(Nations.get_normalized_name(), "cache", digest)
        if directory.is_dir():
            for name in ("training", "testing", "validation"):
                path = directory.joinpath(name).with_suffix(".pt")
                if path.is_file():
                    path.unlink()
            directory.rmdir()

        self.assertFalse(directory.is_dir())
        for name in ("training", "testing", "validation"):
            path = directory.joinpath(name).with_suffix(".pt")
            self.assertFalse(path.is_file())

        _ = _cached_get_dataset("nations", {})
        self.assertTrue(directory.is_dir())
        for name in ("training", "testing", "validation"):
            path = directory.joinpath(name).with_suffix(".pt")
            self.assertTrue(path.is_file())

    def test_serialization(self):
        """Test dataset serialization."""
        dataset = Nations()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            dataset.to_directory_binary(path=path)
            dataset2 = Dataset.from_directory_binary(path=path)
            # TODO: Dataset.__equal__
            self.assertSetEqual(set(dataset.factory_dict.keys()), set(dataset2.factory_dict.keys()))
            for key, tf in dataset.factory_dict.items():
                tf2 = dataset2.factory_dict[key]
                assert tf.num_entities == tf2.num_entities
                assert tf.num_relations == tf2.num_relations
                assert tf.create_inverse_triples == tf2.create_inverse_triples
                assert tf.num_triples == tf2.num_triples
                numpy.testing.assert_array_equal(tf.mapped_triples.numpy(), tf2.mapped_triples.numpy())
