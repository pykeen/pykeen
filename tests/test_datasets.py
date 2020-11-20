# -*- coding: utf-8 -*-

"""Test that datasets can be loaded."""

import os

import pytest

from pykeen.datasets import FB15k, FB15k237, Kinships, Nations, UMLS, WN18, WN18RR, YAGO310
from pykeen.datasets.base import SingleTabbedDataset, TarFileSingleDataset
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from tests import cases, constants


class MockSingleTabbedDataset(SingleTabbedDataset):
    """Mock downloading a single file."""

    def __init__(self, cache_root: str):
        super().__init__(url=..., name=..., cache_root=cache_root)

    def _get_path(self) -> str:
        return NATIONS_TRAIN_PATH


class MockTarFileSingleDataset(TarFileSingleDataset):
    """Mock downloading a tar.gz archive."""

    def __init__(self, cache_root: str):
        super().__init__(url=..., name=..., relative_path='nations/train.txt', cache_root=cache_root)

    def _get_path(self):
        return os.path.join(constants.RESOURCES, 'nations.tar.gz')


class TestSingle(cases.CachedDatasetCase):
    """Test the base classes.

    .. note:: This uses the nations training dataset
    """

    exp_num_entities = 14
    exp_num_relations = 55
    autoloaded_validation = True
    dataset_cls = MockSingleTabbedDataset


class TestTarFileSingle(cases.CachedDatasetCase):
    """Test the base classes.

    .. note:: This uses the nations training dataset
    """

    exp_num_entities = 14
    exp_num_relations = 55
    autoloaded_validation = True
    dataset_cls = MockTarFileSingleDataset


class TestNations(cases.LocalDatasetTestCase):
    """Test the Nations dataset."""

    exp_num_entities = 14
    exp_num_relations = 55
    dataset_cls = Nations


class TestKinships(cases.LocalDatasetTestCase):
    """Test the Nations dataset."""

    exp_num_entities = 104
    exp_num_relations = 25
    dataset_cls = Kinships


class TestUMLS(cases.LocalDatasetTestCase):
    """Test the Nations dataset."""

    exp_num_entities = 135
    exp_num_relations = 46
    dataset_cls = UMLS


@pytest.mark.slow
class TestFB15K(cases.CachedDatasetCase):
    """Test the FB15k dataset."""

    exp_num_entities = 14951
    exp_num_relations = 1345
    dataset_cls = FB15k


@pytest.mark.slow
class TestFB15K237(cases.CachedDatasetCase):
    """Test the FB15k-237 dataset."""

    exp_num_entities = 14505
    exp_num_relations = 237
    dataset_cls = FB15k237


@pytest.mark.slow
class TestWN18(cases.CachedDatasetCase):
    """Test the WN18 dataset."""

    exp_num_entities = 40943
    exp_num_relations = 18
    dataset_cls = WN18


@pytest.mark.slow
class TestWN18RR(cases.CachedDatasetCase):
    """Test the WN18RR dataset."""

    exp_num_entities = 40559
    exp_num_relations = 11
    dataset_cls = WN18RR


@pytest.mark.slow
class TestYAGO310(cases.CachedDatasetCase):
    """Test the YAGO3-10 dataset."""

    exp_num_entities = 123143
    exp_num_relations = 37
    dataset_cls = YAGO310
