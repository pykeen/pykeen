# -*- coding: utf-8 -*-

"""Test that datasets can be loaded."""

import tempfile
import timeit
import unittest
from typing import Type, Union

import pytest

from pykeen.datasets import FB15k, FB15k237, Kinships, Nations, UMLS, WN18, WN18RR, YAGO310
from pykeen.datasets.base import DataSet
from pykeen.triples import TriplesFactory


class _DataSetTestCase:
    """A test case for quickly defining common tests for datasets."""

    #: The expected number of entities
    exp_num_entities: int
    #: The expected number of relations
    exp_num_relations: int
    #: The dataset to test
    dataset: Union[DataSet, Type[DataSet]]

    def setUp(self):
        """Set up the test case."""
        if isinstance(self.dataset, DataSet):
            self.directory = None
            self.dataset = self.dataset
        else:
            self.directory = tempfile.TemporaryDirectory()
            self.dataset = self.dataset(cache_root=self.directory.name)

    def tearDown(self) -> None:
        """Tear down the test case."""
        if self.directory is not None:
            self.directory.cleanup()

    def test_dataset(self):
        """Generic test for datasets."""
        self.assertIsInstance(self.dataset, DataSet)

        # Not loaded
        assert self.dataset._training is None
        assert self.dataset._testing is None
        assert self.dataset._validation is None
        assert not self.dataset._loaded
        assert not self.dataset._loaded_validation

        # Load
        self.dataset._load()
        assert isinstance(self.dataset.training, TriplesFactory)
        assert isinstance(self.dataset.testing, TriplesFactory)
        assert self.dataset._loaded

        assert not self.dataset._loaded_validation
        self.dataset._load_validation()
        assert isinstance(self.dataset.validation, TriplesFactory)

        assert self.dataset._training is not None
        assert self.dataset._testing is not None
        assert self.dataset._validation is not None
        assert self.dataset._loaded
        assert self.dataset._loaded_validation

        assert self.dataset.num_entities == self.exp_num_entities
        assert self.dataset.num_relations == self.exp_num_relations

        # Test caching
        start = timeit.default_timer()
        self.dataset.training
        end = timeit.default_timer()
        assert (end - start) < 1.0e-02


class TestNations(_DataSetTestCase, unittest.TestCase):
    """Test the Nations dataset."""

    exp_num_entities = 14
    exp_num_relations = 55
    dataset = Nations()


class TestKinships(_DataSetTestCase, unittest.TestCase):
    """Test the Nations dataset."""

    exp_num_entities = 104
    exp_num_relations = 25
    dataset = Kinships()


class TestUMLS(_DataSetTestCase, unittest.TestCase):
    """Test the Nations dataset."""

    exp_num_entities = 135
    exp_num_relations = 46
    dataset = UMLS()


@pytest.mark.slow
class TestFB15k(_DataSetTestCase, unittest.TestCase):
    """Test the FB15k dataset."""

    exp_num_entities = 14951
    exp_num_relations = 1345
    dataset = FB15k


@pytest.mark.slow
class TestFB15k237(_DataSetTestCase, unittest.TestCase):
    """Test the FB15k-237 dataset."""

    exp_num_entities = 14505
    exp_num_relations = 237
    dataset = FB15k237


@pytest.mark.slow
class TestWN18(_DataSetTestCase, unittest.TestCase):
    """Test the WN18 dataset."""

    exp_num_entities = 40943
    exp_num_relations = 18
    dataset = WN18


@pytest.mark.slow
class TestWN18RR(_DataSetTestCase, unittest.TestCase):
    """Test the WN18RR dataset."""

    exp_num_entities = 40559
    exp_num_relations = 11
    dataset = WN18RR


@pytest.mark.slow
class TestYAGO310(_DataSetTestCase, unittest.TestCase):
    """Test the YAGO3-10 dataset."""

    exp_num_entities = 123143
    exp_num_relations = 37
    dataset = YAGO310
