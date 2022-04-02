# -*- coding: utf-8 -*-

"""Test that datasets can be loaded."""

import pathlib
import unittest
from io import BytesIO
from urllib.request import urlopen

from pykeen.datasets import Kinships, Nations, dataset_resolver
from pykeen.datasets.base import (
    PackedZipRemoteDataset,
    SingleTabbedDataset,
    TarFileRemoteDataset,
    TarFileSingleDataset,
    UnpackedRemoteDataset,
)
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH, NATIONS_VALIDATE_PATH
from tests import cases, constants


class TestAnnotated(unittest.TestCase):
    """Test all datasets are annotated."""

    def test_annotated(self):
        """Check :func:`pykeen.utils_docs.with_structured_docstr`` was properly applied ot all datasets."""
        for name, cls in sorted(dataset_resolver.lookup_dict.items()):
            with self.subTest(name=name):
                try:
                    docdata = cls.__docdata__
                except AttributeError:
                    self.fail("missing __docdata__")
                self.assertIn("name", docdata)
                self.assertIsInstance(docdata["name"], str)
                self.assertIn("statistics", docdata)
                self.assertIn("citation", docdata)

                # Check minimal statistics
                for k in ("entities", "relations", "triples", "training", "testing", "validation"):
                    self.assertIn(k, docdata["statistics"], msg=f"statistics are missing {k}")
                    self.assertIsInstance(docdata["statistics"][k], int)

                # Check either a github link or author/publication information is given
                citation = docdata["citation"]
                self.assertTrue(
                    ("author" in citation and "link" in citation and "year" in citation) or "github" in citation,
                )

            signature = dataset_resolver.signature(cls)
            random_state_param = signature.parameters.get("random_state")
            if random_state_param is not None:
                self.assertEqual(0, random_state_param.default)


class MockSingleTabbedDataset(SingleTabbedDataset):
    """Mock downloading a single file."""

    def __init__(self, cache_root: str):
        super().__init__(url=..., name=..., cache_root=cache_root)

    def _get_path(self) -> str:
        return NATIONS_TRAIN_PATH


class MockTarFileSingleDataset(TarFileSingleDataset):
    """Mock downloading a tar.gz archive with a single file."""

    def __init__(self, cache_root: str):
        super().__init__(
            url=...,
            name=...,
            relative_path="nations/train.txt",
            cache_root=cache_root,
        )

    def _get_path(self) -> str:
        return constants.RESOURCES.joinpath("nations.tar.gz")


class MockTarFileRemoteDataset(TarFileRemoteDataset):
    """Mock downloading a tar.gz archive with three pre-stratified files."""

    def __init__(self, cache_root: str):
        super().__init__(
            url=constants.RESOURCES.joinpath("nations.tar.gz").as_uri(),
            cache_root=cache_root,
            relative_testing_path=pathlib.PurePath("nations", "test.txt"),
            relative_training_path=pathlib.PurePath("nations", "train.txt"),
            relative_validation_path=pathlib.PurePath("nations", "valid.txt"),
        )

    def _get_bytes(self) -> BytesIO:
        return BytesIO(urlopen(self.url).read())  # noqa:S310


class MockUnpackedRemoteDataset(UnpackedRemoteDataset):
    """Mock downloading three pre-stratified files."""

    def __init__(self):
        super().__init__(
            training_url=NATIONS_TRAIN_PATH.as_uri(),
            testing_url=NATIONS_TEST_PATH.as_uri(),
            validation_url=NATIONS_VALIDATE_PATH.as_uri(),
        )


class MockZipFileRemoteDataset(PackedZipRemoteDataset):
    """Mock downloading a zip archive with three pre-stratified files."""

    def __init__(self, cache_root: str):
        super().__init__(
            url=constants.RESOURCES.joinpath("nations.zip").as_uri(),
            cache_root=cache_root,
            relative_testing_path=pathlib.PurePath("nations", "test.txt"),
            relative_training_path=pathlib.PurePath("nations", "train.txt"),
            relative_validation_path=pathlib.PurePath("nations", "valid.txt"),
        )


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


class TestUnpackedRemote(cases.LocalDatasetTestCase):
    """Test the loading an uncompressed, pre-stratified dataset with Nations as the example."""

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1992
    dataset_cls = MockUnpackedRemoteDataset


class TestZipFileRemote(cases.CachedDatasetCase):
    """Test the loading an zip compressed, pre-stratified dataset with Nations as the example."""

    exp_num_entities = 14
    exp_num_relations = 55
    exp_num_triples = 1992
    dataset_cls = MockZipFileRemoteDataset
