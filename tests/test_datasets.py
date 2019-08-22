# coding=utf-8

"""Test that datasets can be loaded."""

import timeit
import unittest

import pytest

from poem.datasets import DataSet, fb15k, fb15k237, kinship, nations, umls, wn18, wn18rr, yago3_10


class _DataSetTestCase:
    """A test case for quickly defining common tests for datasets."""

    #: The expected number of entities
    exp_num_entities: int
    #: The expected number of relations
    exp_num_relations: int
    #: The dataset to test
    dataset: DataSet

    def test_dataset(self):
        """Generic test for datasets."""
        # Not loaded
        assert self.dataset.training is None
        assert self.dataset.testing is None
        assert self.dataset.validation is None
        assert not self.dataset._loaded

        # Load
        loaded_dataset = self.dataset.load()
        assert loaded_dataset is self.dataset

        assert self.dataset.training is not None
        assert self.dataset.testing is not None
        assert self.dataset.validation is not None
        assert self.dataset._loaded

        assert self.dataset.num_entities == self.exp_num_entities
        assert self.dataset.num_relations == self.exp_num_relations

        # Test caching
        start = timeit.default_timer()
        self.dataset.load()
        end = timeit.default_timer()
        assert (end - start) < 1.0e-02


class TestNations(_DataSetTestCase, unittest.TestCase):
    """Test the Nations dataset."""

    exp_num_entities = 14
    exp_num_relations = 55
    dataset = nations


class TestKinship(_DataSetTestCase, unittest.TestCase):
    """Test the Nations dataset."""

    exp_num_entities = 104
    exp_num_relations = 25
    dataset = kinship


class TestUMLS(_DataSetTestCase, unittest.TestCase):
    """Test the Nations dataset."""

    exp_num_entities = 135
    exp_num_relations = 46
    dataset = umls


@pytest.mark.slow
class TestFB15k(_DataSetTestCase, unittest.TestCase):
    """Test the FB15k dataset."""

    exp_num_entities = 14951
    exp_num_relations = 1345
    dataset = fb15k


@pytest.mark.slow
class TestFB15k237(_DataSetTestCase, unittest.TestCase):
    """Test the FB15k-237 dataset."""

    exp_num_entities = 14505
    exp_num_relations = 237
    dataset = fb15k237


@pytest.mark.slow
class TestWN18(_DataSetTestCase, unittest.TestCase):
    """Test the WN18 dataset."""

    exp_num_entities = 40943
    exp_num_relations = 18
    dataset = wn18


@pytest.mark.slow
class TestWN18RR(_DataSetTestCase, unittest.TestCase):
    """Test the WN18RR dataset."""

    exp_num_entities = 40559
    exp_num_relations = 11
    dataset = wn18rr


@pytest.mark.slow
class TestYAGO310(_DataSetTestCase, unittest.TestCase):
    """Test the YAGO3-10 dataset."""

    exp_num_entities = 123143
    exp_num_relations = 37
    dataset = yago3_10
