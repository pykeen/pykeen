"""Tests for the dataset sources."""

import pathlib
import tempfile
import unittest

import pytest

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
