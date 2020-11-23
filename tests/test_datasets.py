# -*- coding: utf-8 -*-

"""Test that datasets can be loaded."""

import os
from io import BytesIO

import pytest

from pykeen.datasets import FB15k, FB15k237, Kinships, Nations, UMLS, WN18, WN18RR, YAGO310
from pykeen.datasets.base import SingleTabbedDataset, TarFileRemoteDataSet, TarFileSingleDataset
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from tests import cases, constants


class MockSingleTabbedDataset(SingleTabbedDataset):
    """Mock downloading a single file."""

    def __init__(self, cache_root: str):
        super().__init__(url=..., name=..., cache_root=cache_root)

    def _get_path(self) -> str:
        return NATIONS_TRAIN_PATH


class MockTarFileSingleDataset(TarFileSingleDataset):
    """Mock downloading a tar.gz archive with a single file."""

    def __init__(self, cache_root: str):
        super().__init__(url=..., name=..., relative_path='nations/train.txt', cache_root=cache_root)

    def _get_path(self) -> str:
        return os.path.join(constants.RESOURCES, 'nations.tar.gz')


class MockTarFileRemoteDataset(TarFileRemoteDataSet):
    """Mock downloading a tar.gz archive with three pre-stratified files."""

    def __init__(self, cache_root: str):
        super().__init__(
            url=...,
            cache_root=cache_root,
            relative_testing_path='nations/test.txt',
            relative_training_path='nations/train.txt',
            relative_validation_path='nations/valid.txt',
        )

    def _get_bytes(self) -> BytesIO:
        with open(os.path.join(constants.RESOURCES, 'nations.tar.gz'), 'rb') as file:
            return BytesIO(file.read())


class TestSingle(cases.CachedDatasetCase):
    """Test the base classes.

    .. note:: This uses the nations training dataset
    """

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1592  # because only loading training set from Nations
    exp_num_triples_tolerance = 5
    autoloaded_validation = True
    dataset_cls = MockSingleTabbedDataset


class TestTarFileSingle(cases.CachedDatasetCase):
    """Test the base classes.

    .. note:: This uses the nations training dataset
    """

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1592  # because only loading training set from Nations
    exp_num_triples_tolerance = 5
    autoloaded_validation = True
    dataset_cls = MockTarFileSingleDataset


class TestTarRemote(cases.CachedDatasetCase):
    """Test the :class:`pykeen.datasets.base.TarFileRemoteDataSet` class."""

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1992
    dataset_cls = MockTarFileRemoteDataset


class TestNations(cases.LocalDatasetTestCase):
    """Test the Nations dataset."""

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1992
    dataset_cls = Nations


class TestKinships(cases.LocalDatasetTestCase):
    """Test the Nations dataset."""

    exp_num_entities = 104
    exp_num_relations = 25
    exp_num_triples = 10686
    dataset_cls = Kinships


class TestUMLS(cases.LocalDatasetTestCase):
    """Test the Nations dataset."""

    exp_num_entities = 135
    exp_num_relations = 46
    exp_num_triples = 6529
    dataset_cls = UMLS


@pytest.mark.slow
class TestFB15K(cases.CachedDatasetCase):
    """Test the FB15k dataset."""

    exp_num_entities = 14951
    exp_num_relations = 1345
    exp_num_triples = 592_213
    dataset_cls = FB15k


@pytest.mark.slow
class TestFB15K237(cases.CachedDatasetCase):
    """Test the FB15k-237 dataset."""

    exp_num_entities = 14505
    exp_num_relations = 237
    exp_num_triples = 310_079
    dataset_cls = FB15k237


@pytest.mark.slow
class TestWN18(cases.CachedDatasetCase):
    """Test the WN18 dataset."""

    exp_num_entities = 40943
    exp_num_relations = 18
    exp_num_triples = 151_442
    dataset_cls = WN18


@pytest.mark.slow
class TestWN18RR(cases.CachedDatasetCase):
    """Test the WN18RR dataset."""

    exp_num_entities = 40559
    exp_num_relations = 11
    exp_num_triples = 92583
    dataset_cls = WN18RR


@pytest.mark.slow
class TestYAGO310(cases.CachedDatasetCase):
    """Test the YAGO3-10 dataset."""

    exp_num_entities = 123_143
    exp_num_relations = 37
    exp_num_triples = 1_089_000
    dataset_cls = YAGO310
