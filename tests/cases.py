# -*- coding: utf-8 -*-

"""Test cases for PyKEEN."""

import tempfile
import timeit
import unittest
from typing import ClassVar, Optional, Type

from pykeen.datasets.base import LazyDataSet
from pykeen.triples import TriplesFactory


class DatasetTestCase(unittest.TestCase):
    """A test case for quickly defining common tests for datasets."""

    #: The expected number of entities
    exp_num_entities: ClassVar[int]
    #: The expected number of relations
    exp_num_relations: ClassVar[int]
    #: The expected number of triples
    exp_num_triples: ClassVar[int]
    #: The tolerance on expected number of triples, for randomized situations
    exp_num_triples_tolerance: ClassVar[Optional[int]] = None

    #: The dataset to test
    dataset_cls: ClassVar[Type[LazyDataSet]]
    #: The instantiated dataset
    dataset: LazyDataSet

    #: Should the validation be assumed to have been loaded with train/test?
    autoloaded_validation: ClassVar[bool] = False

    def test_dataset(self):
        """Generic test for datasets."""
        self.assertIsInstance(self.dataset, LazyDataSet)

        # Not loaded
        self.assertIsNone(self.dataset._training)
        self.assertIsNone(self.dataset._testing)
        self.assertIsNone(self.dataset._validation)
        self.assertFalse(self.dataset._loaded)
        self.assertFalse(self.dataset._loaded_validation)

        # Load
        try:
            self.dataset._load()
        except (EOFError, IOError):
            self.skipTest('Problem with connection. Try this test again later.')

        self.assertIsInstance(self.dataset.training, TriplesFactory)
        self.assertIsInstance(self.dataset.testing, TriplesFactory)
        self.assertTrue(self.dataset._loaded)

        if self.autoloaded_validation:
            self.assertTrue(self.dataset._loaded_validation)
        else:
            self.assertFalse(self.dataset._loaded_validation)
            self.dataset._load_validation()

        self.assertIsInstance(self.dataset.validation, TriplesFactory)

        self.assertIsNotNone(self.dataset._training)
        self.assertIsNotNone(self.dataset._testing)
        self.assertIsNotNone(self.dataset._validation)
        self.assertTrue(self.dataset._loaded)
        self.assertTrue(self.dataset._loaded_validation)

        self.assertEqual(self.dataset.num_entities, self.exp_num_entities)
        self.assertEqual(self.dataset.num_relations, self.exp_num_relations)

        num_triples = sum(
            triples_factory.num_triples for
            triples_factory in (self.dataset._training, self.dataset._testing, self.dataset._validation)
        )
        if self.exp_num_triples_tolerance is None:
            self.assertEqual(self.exp_num_triples, num_triples)
        else:
            self.assertAlmostEqual(self.exp_num_triples, num_triples, delta=self.exp_num_triples_tolerance)

        # Test caching
        start = timeit.default_timer()
        _ = self.dataset.training
        end = timeit.default_timer()
        # assert (end - start) < 1.0e-02
        self.assertAlmostEqual(start, end, delta=1.0e-02, msg='Caching should have made this operation fast')


class LocalDatasetTestCase(DatasetTestCase):
    """A test case for datasets that don't need a cache directory."""

    def setUp(self):
        """Set up the test case."""
        self.dataset = self.dataset_cls()


class CachedDatasetCase(DatasetTestCase):
    """A test case for datasets that need a cache directory."""

    #: The directory, if there is caching
    directory: Optional[tempfile.TemporaryDirectory]

    def setUp(self):
        """Set up the test with a temporary cache directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.dataset = self.dataset_cls(cache_root=self.directory.name)

    def tearDown(self) -> None:
        """Tear down the test case by cleaning up the temporary cache directory."""
        self.directory.cleanup()
