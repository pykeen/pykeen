"""Tests for the dataset sources and loaders."""

import pathlib
import tempfile
import unittest

import pytest

from pykeen.datasets.loaders import INDUCTIVE_PLAN, TRANSDUCTIVE_PLAN, PreSplitLoader
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH, NATIONS_VALIDATE_PATH
from pykeen.datasets.sources import (
    ArchiveSource,
    LocalSource,
    RemoteSource,
    TarArchiveSource,
    ZipArchiveSource,
)
from tests import constants


class TestLocalSource(unittest.TestCase):
    """Tests for :class:`pykeen.datasets.sources.LocalSource`."""

    def test_paths(self):
        """Test that the paths are passed through."""
        source = LocalSource(training=NATIONS_TRAIN_PATH, testing=str(NATIONS_TEST_PATH))
        assert source.paths() == {"training": NATIONS_TRAIN_PATH, "testing": NATIONS_TEST_PATH}

    def test_none_is_dropped(self):
        """Test that a ``None`` path is dropped rather than mapped to ``None``."""
        source = LocalSource(training=NATIONS_TRAIN_PATH, testing=NATIONS_TEST_PATH, validation=None)
        assert "validation" not in source.paths()


class TestRemoteSource(unittest.TestCase):
    """Tests for :class:`pykeen.datasets.sources.RemoteSource`."""

    def setUp(self):
        """Set up a temporary cache directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.cache_root = pathlib.Path(self.directory.name)

    def tearDown(self):
        """Clean up the temporary cache directory."""
        self.directory.cleanup()

    def _make(self, **kwargs) -> RemoteSource:
        return RemoteSource(
            urls={
                "training": NATIONS_TRAIN_PATH.as_uri(),
                "testing": NATIONS_TEST_PATH.as_uri(),
                "validation": NATIONS_VALIDATE_PATH.as_uri(),
            },
            cache_root=self.cache_root,
            **kwargs,
        )

    def test_expected_paths_do_not_download(self):
        """Test that asking where the files will be does not download them."""
        source = self._make()
        paths = source.expected_paths()
        assert set(paths) == {"training", "testing", "validation"}
        assert not any(path.is_file() for path in paths.values())

    def test_download(self):
        """Test that materializing downloads the files."""
        source = self._make()
        paths = source.paths()
        assert all(path.is_file() for path in paths.values())
        assert paths["training"] == self.cache_root.joinpath("train.txt")

    def test_sub_directories(self):
        """Test that the per-key sub-directories are used."""
        source = self._make(sub_directories={"training": "a", "testing": "b", "validation": "b"})
        paths = source.paths()
        assert paths["training"] == self.cache_root.joinpath("a", "train.txt")
        assert paths["testing"] == self.cache_root.joinpath("b", "test.txt")
        assert all(path.is_file() for path in paths.values())


class ArchiveSourceTests:
    """A base test case for archive sources.

    .. note::

        This is a plain mixin rather than a :class:`unittest.TestCase`, so that pytest does not collect it.
    """

    #: The source class under test
    source_cls: type[ArchiveSource]
    #: The name of the archive inside the test resources
    archive_name: str

    def setUp(self):
        """Set up a temporary cache directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.cache_root = pathlib.Path(self.directory.name)
        self.archive_path = constants.RESOURCES.joinpath(self.archive_name)

    def tearDown(self):
        """Clean up the temporary cache directory."""
        self.directory.cleanup()

    def _make(self, **kwargs) -> ArchiveSource:
        return self.source_cls(
            members={"training": pathlib.PurePath("nations", "train.txt")},
            cache_root=self.cache_root,
            archive_path=self.archive_path,
            **kwargs,
        )

    def test_extract_member(self):
        """Test that only the requested member is extracted."""
        source = self._make()
        paths = source.paths()
        assert paths["training"] == self.cache_root.joinpath("nations", "train.txt")
        assert paths["training"].is_file()
        assert not self.cache_root.joinpath("nations", "test.txt").is_file()

    def test_extract_all(self):
        """Test that the whole archive is unpacked on request."""
        source = self._make(extract_all=True)
        source.materialize()
        assert self.cache_root.joinpath("nations", "test.txt").is_file()

    def test_no_url_no_archive(self):
        """Test that a missing archive without a URL is reported."""
        source = self.source_cls(
            members={"training": pathlib.PurePath("nations", "train.txt")},
            cache_root=self.cache_root,
            name="does-not-exist",
        )
        with pytest.raises(ValueError, match="must specify url"):
            source.paths()


class TestTarArchiveSource(ArchiveSourceTests, unittest.TestCase):
    """Tests for :class:`pykeen.datasets.sources.TarArchiveSource`."""

    source_cls = TarArchiveSource
    archive_name = "nations.tar.gz"


class TestZipArchiveSource(ArchiveSourceTests, unittest.TestCase):
    """Tests for :class:`pykeen.datasets.sources.ZipArchiveSource`."""

    source_cls = ZipArchiveSource
    archive_name = "nations.zip"


class TestArchiveSourceValidation(unittest.TestCase):
    """Tests for the construction-time validation of archive sources."""

    def test_missing_name(self):
        """Test that a source without any way to locate its archive is rejected."""
        with pytest.raises(ValueError, match="at least one of"):
            TarArchiveSource(members={}, cache_root=pathlib.Path())


class TestPreSplitLoader(unittest.TestCase):
    """Tests for :class:`pykeen.datasets.loaders.PreSplitLoader`."""

    def test_transductive_plan(self):
        """Test that the evaluation splits share the training index."""
        loader = PreSplitLoader(
            source=LocalSource(
                training=NATIONS_TRAIN_PATH, testing=NATIONS_TEST_PATH, validation=NATIONS_VALIDATE_PATH
            ),
            create_inverse_triples=True,
        )
        factories = loader.load()
        assert set(factories) == {"training", "testing", "validation"}
        training = factories["training"]
        # inverse triples are only created for training; evaluation handles them itself
        assert training.create_inverse_triples
        for key in ("testing", "validation"):
            assert not factories[key].create_inverse_triples
            assert factories[key].entity_to_id == training.entity_to_id
            assert factories[key].relation_to_id == training.relation_to_id

    def test_missing_optional_split(self):
        """Test that an absent validation split is simply omitted."""
        loader = PreSplitLoader(source=LocalSource(training=NATIONS_TRAIN_PATH, testing=NATIONS_TEST_PATH))
        assert set(loader.load()) == {"training", "testing"}

    def test_inductive_plan(self):
        """Test that the inductive plan shares relations with training, but entities with the inference graph."""
        # note: the actual triples do not matter here, only which index each factory inherits
        loader = PreSplitLoader(
            source=LocalSource(
                transductive_training=NATIONS_TRAIN_PATH,
                inductive_inference=NATIONS_TRAIN_PATH,
                inductive_testing=NATIONS_TEST_PATH,
                inductive_validation=NATIONS_VALIDATE_PATH,
            ),
            plan=INDUCTIVE_PLAN,
            create_inverse_triples=True,
        )
        factories = loader.load()
        transductive_training = factories["transductive_training"]
        inference = factories["inductive_inference"]
        assert inference.relation_to_id == transductive_training.relation_to_id
        assert inference.create_inverse_triples
        for key in ("inductive_testing", "inductive_validation"):
            assert not factories[key].create_inverse_triples
            assert factories[key].entity_to_id == inference.entity_to_id
            assert factories[key].relation_to_id == inference.relation_to_id

    def test_plans_are_topologically_ordered(self):
        """Test that a split never inherits an index from a split which is built later."""
        for plan in (TRANSDUCTIVE_PLAN, INDUCTIVE_PLAN):
            seen: set[str] = set()
            for key, spec in plan.items():
                for source_key in (spec.entity_index_from, spec.relation_index_from):
                    assert source_key is None or source_key in seen, f"{key} inherits from a later split"
                seen.add(key)
