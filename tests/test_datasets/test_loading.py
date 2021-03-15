# -*- coding: utf-8 -*-

"""Test that datasets can be loaded."""

import os
import unittest
from io import BytesIO

import pytest

from pykeen.datasets import FB15k237, Kinships, Nations, datasets
from pykeen.datasets.base import SingleTabbedDataset, TarFileRemoteDataset, TarFileSingleDataset
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
from tests import cases, constants


class TestAnnotated(unittest.TestCase):
    """Test all datasets are annotated."""

    def test_annotated(self):
        """Check :func:`pykeen.utils_docs.with_structured_docstr`` was properly applied ot all datasets."""
        for name, cls in sorted(datasets.items()):
            with self.subTest(name=name):
                try:
                    docdata = cls.__docdata__
                except AttributeError:
                    self.fail('missing __docdata__')
                self.assertIn('name', docdata)
                self.assertIsInstance(docdata['name'], str)
                self.assertIn('statistics', docdata)
                self.assertIn('citation', docdata)

                # Check minimal statistics
                for k in ('entities', 'relations', 'triples'):
                    self.assertIn(k, docdata['statistics'], msg=f'statistics are missing {k}')
                    self.assertIsInstance(docdata['statistics'][k], int)

                # Check statistics for pre-stratified datasets
                if not docdata.get('single'):
                    for k in ('training', 'testing', 'validation'):
                        self.assertIn(k, docdata['statistics'])
                        self.assertIsInstance(docdata['statistics'][k], int)

                # Check either a github link or author/publication information is given
                citation = docdata['citation']
                self.assertTrue(
                    ('author' in citation and 'link' in citation and 'year' in citation)
                    or 'github' in citation,
                )


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


class MockTarFileRemoteDataset(TarFileRemoteDataset):
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
    """Test the :class:`pykeen.datasets.base.TarFileRemoteDataset` class."""

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1992
    dataset_cls = MockTarFileRemoteDataset


class TestPathDatasetTriples(cases.LocalDatasetTestCase):
    """Test the :class:`pykeen.datasets.PathDataset` with inverse triples."""

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1992
    dataset_cls = Nations

    def test_create_inverse_triples(self):
        """Verify that inverse triples are only created in the training factory."""
        dataset = Nations(create_inverse_triples=True)
        assert dataset.training.create_inverse_triples
        assert not dataset.testing.create_inverse_triples
        assert not dataset.validation.create_inverse_triples


class TestPathDataset(cases.LocalDatasetTestCase):
    """Test the :class:`pykeen.datasets.PathDataset` without inverse triples."""

    exp_num_entities = 104
    exp_num_relations = 25
    exp_num_triples = 10686
    dataset_cls = Kinships


# TestFB15K237 is a stand-in to test the ZipFileRemoteDataset

@pytest.mark.slow
class TestFB15K237(cases.CachedDatasetCase):
    """Test the FB15k-237 dataset."""

    exp_num_entities = 14505
    exp_num_relations = 237
    exp_num_triples = 310_079
    dataset_cls = FB15k237
