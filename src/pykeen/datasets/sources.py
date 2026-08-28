"""Sources resolve the files backing a dataset to local paths.

A :class:`Source` encapsulates *where the data lives* -- a local directory, one URL per split, or members of a
remote archive -- and is deliberately ignorant of how those files are turned into triples factories. Turning files
into factories is the job of a :class:`~pykeen.datasets.loaders.Loader`.

Keeping the two apart means that a new location, e.g., a new archive format, does not require re-implementing the
loading logic, and vice versa. The dataset classes in :mod:`pykeen.datasets.base` are thin wrappers which pick a
source and a loader.

Every source distinguishes the paths it *will* have, cf. :meth:`Source.expected_paths`, from the paths it *does*
have, cf. :meth:`Source.paths`. Only the latter triggers a download, which keeps lazy datasets lazy while still
allowing them to report where their data lives.
"""

from __future__ import annotations

import logging
import pathlib
import tarfile
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pystow.utils import download, name_from_url

__all__ = [
    "Source",
    "LocalSource",
    "RemoteSource",
    "ArchiveSource",
    "TarArchiveSource",
    "ZipArchiveSource",
]

logger = logging.getLogger(__name__)


class Source(ABC):
    """A resolver from logical file keys to local paths."""

    @abstractmethod
    def expected_paths(self) -> Mapping[str, pathlib.Path]:
        """Return where the files are (or will be), *without* downloading anything.

        Keys which the source cannot provide are omitted rather than mapped to ``None``. This is how an absent
        optional split, e.g., validation, is expressed.

        :returns: A mapping from logical file key, e.g., ``"training"``, to a local path.
        """
        raise NotImplementedError

    def materialize(self) -> None:
        """Ensure the files are present locally, downloading and extracting them if necessary.

        The default does nothing, which is correct for sources whose files are always present.
        """
        return

    def paths(self) -> Mapping[str, pathlib.Path]:
        """Materialize the files and return the mapping from key to local path.

        :returns: A mapping from logical file key, e.g., ``"training"``, to an existing local path.
        """
        self.materialize()
        return self.expected_paths()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}()"


class LocalSource(Source):
    """Files which are already present on the local file system."""

    def __init__(self, **paths: None | str | pathlib.Path) -> None:
        """Initialize the source.

        :param paths: The local paths, keyed by their logical file key. ``None`` values are dropped.
        """
        self._paths = {key: pathlib.Path(path) for key, path in paths.items() if path is not None}

    # docstr-coverage: inherited
    def expected_paths(self) -> Mapping[str, pathlib.Path]:  # noqa: D102
        return self._paths

    def __repr__(self) -> str:  # noqa: D105
        inner = ", ".join(f'{key}="{path}"' for key, path in self._paths.items())
        return f"{self.__class__.__name__}({inner})"


class RemoteSource(Source):
    """One separately downloaded file per key."""

    def __init__(
        self,
        urls: Mapping[str, str],
        cache_root: pathlib.Path,
        *,
        sub_directories: Mapping[str, str] | None = None,
        force: bool = False,
        download_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the source.

        :param urls: The URL for each logical file key.
        :param cache_root: The directory into which the files are downloaded.
        :param sub_directories: An optional sub-directory of ``cache_root`` per key. Used by datasets which keep,
            e.g., the training and the inference part in separate directories.
        :param force: Whether to re-download files which are already present.
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`.
        """
        self.urls = dict(urls)
        self.cache_root = cache_root
        self.sub_directories = dict(sub_directories or {})
        self.force = force
        download_kwargs = {} if download_kwargs is None else dict(download_kwargs)
        download_kwargs.setdefault("backend", "urllib")
        self.download_kwargs = download_kwargs

    # docstr-coverage: inherited
    def expected_paths(self) -> Mapping[str, pathlib.Path]:  # noqa: D102
        result = {}
        for key, url in self.urls.items():
            directory = self.cache_root
            sub_directory = self.sub_directories.get(key)
            if sub_directory is not None:
                directory = directory.joinpath(sub_directory)
            result[key] = directory.joinpath(name_from_url(url))
        return result

    # docstr-coverage: inherited
    def materialize(self) -> None:  # noqa: D102
        for key, path in self.expected_paths().items():
            if not self.force and path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            download(url=self.urls[key], path=path, force=self.force, **self.download_kwargs)

    def __repr__(self) -> str:  # noqa: D105
        inner = ", ".join(f'{key}="{url}"' for key, url in self.urls.items())
        return f"{self.__class__.__name__}({inner})"


class ArchiveSource(Source):
    """Members of a single archive, which is downloaded once and extracted into the cache root."""

    def __init__(
        self,
        members: Mapping[str, str | pathlib.PurePath],
        cache_root: pathlib.Path,
        *,
        url: str | None = None,
        name: str | None = None,
        archive_path: pathlib.Path | None = None,
        extract_all: bool = False,
        force: bool = False,
        download_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the source.

        :param members: The path *inside* the archive for each logical file key. After extraction, the file is
            found at ``cache_root / member``.
        :param cache_root: The directory into which the archive is downloaded and extracted.
        :param url: The URL from which to download the archive. May be ``None`` if the archive is already present.
        :param name: The file name of the archive. Defaults to the last part of the URL.
        :param archive_path: An explicit location for the archive, overriding ``cache_root / name``.
        :param extract_all: Whether to unpack the whole archive rather than only the requested members. This is
            appropriate for archives which contain little besides the dataset itself, and is more robust against
            member names which do not exactly match the requested ones.
        :param force: Whether to re-download and re-extract even if the files are already present.
        :param download_kwargs: Keyword arguments to pass to :func:`pystow.utils.download`.

        :raises ValueError: If neither a URL, a name, nor an archive path is given.
        """
        if name is None and archive_path is None:
            if url is None:
                raise ValueError("must give at least one of name, archive_path, or url")
            name = name_from_url(url)
        self.members = {key: pathlib.PurePath(member) for key, member in members.items()}
        self.cache_root = cache_root
        self.url = url
        self.name = name
        self._archive_path = archive_path
        self.extract_all = extract_all
        self.force = force
        download_kwargs = {} if download_kwargs is None else dict(download_kwargs)
        download_kwargs.setdefault("backend", "urllib")
        self.download_kwargs = download_kwargs

    @property
    def archive_path(self) -> pathlib.Path:
        """The local path of the archive."""
        if self._archive_path is not None:
            return self._archive_path
        assert self.name is not None
        return self.cache_root.joinpath(self.name)

    def _ensure_archive(self) -> pathlib.Path:
        """Download the archive if it is not present yet.

        :returns: The local path of the archive.

        :raises ValueError: If the archive is missing and there is no URL to download it from.
        """
        path = self.archive_path
        if path.is_file() and not self.force:
            return path
        if self.url is None:
            raise ValueError(f"must specify url to download from since path does not exist: {path}")
        logger.info("downloading data from %s to %s", self.url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        download(url=self.url, path=path, force=self.force, **self.download_kwargs)
        return path

    @abstractmethod
    def _extract(self, archive_path: pathlib.Path) -> None:
        """Extract the archive into :attr:`cache_root`.

        :param archive_path: The local path of the archive.
        """
        raise NotImplementedError

    # docstr-coverage: inherited
    def expected_paths(self) -> Mapping[str, pathlib.Path]:  # noqa: D102
        return {key: self.cache_root.joinpath(member) for key, member in self.members.items()}

    # docstr-coverage: inherited
    def materialize(self) -> None:  # noqa: D102
        if not self.force and all(path.is_file() for path in self.expected_paths().values()):
            return
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._extract(archive_path=self._ensure_archive())
        logger.info("extracted %s to %s", self.archive_path, self.cache_root)

    def __repr__(self) -> str:  # noqa: D105
        return f'{self.__class__.__name__}(url="{self.url}", archive_path="{self.archive_path}")'


class TarArchiveSource(ArchiveSource):
    """Members of a tar archive."""

    # docstr-coverage: inherited
    def _extract(self, archive_path: pathlib.Path) -> None:  # noqa: D102
        with tarfile.open(archive_path) as tar_file:
            if self.extract_all:
                tar_file.extractall(path=self.cache_root)  # noqa:S202
                return
            for member in self.members.values():
                # tarfile does not like pathlib
                tar_file.extract(str(member), self.cache_root)


class ZipArchiveSource(ArchiveSource):
    """Members of a zip archive."""

    # docstr-coverage: inherited
    def _extract(self, archive_path: pathlib.Path) -> None:  # noqa: D102
        with zipfile.ZipFile(file=archive_path) as zip_file:
            if self.extract_all:
                zip_file.extractall(path=self.cache_root)  # noqa:S202
                return
            for member in self.members.values():
                # paths inside a zip file always use POSIX separators, even on Windows
                zip_file.extract(member.as_posix(), self.cache_root)
