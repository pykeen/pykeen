"""A generic source wrapper."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import IO

from pystow.utils import (
    ArchiveType,
    DownloadKwargs,
    download,
    download_from_google,
    open_archive,
    safe_open,
)

__all__ = [
    "ArchivedSource",
    "EnsuredSource",
    "GoogleArchivedSource",
    "GoogleSource",
    "RemoteArchivedSource",
    "RemoteSimpleSource",
    "RemoteSource",
    "SimpleSource",
    "Source",
]


@dataclass
class Source(ABC):
    """A configuration for a data resource."""

    #: The local path where the resource is stored on disk.
    path: Path

    @abstractmethod
    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Open the source as a file object."""
        raise NotImplementedError


class EnsuredSource(Source, ABC):
    """A source that needs to be ensured before opening."""

    @abstractmethod
    def ensure(self) -> None:
        """Ensure the remote dataset is downloaded."""

    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Download the file and open it using polymorphism."""
        self.ensure()
        with super().open() as file:  # type: ignore[safe-super]
            yield file


@dataclass
class RemoteSource(EnsuredSource):
    """A mixin source for remote sources that must be downloaded before opening."""

    #: The remove location of the file.
    url: str
    force: bool = False
    download_kwargs: DownloadKwargs | None = None

    def ensure(self) -> None:
        """Ensure the remote dataset is downloaded."""
        download(self.url, self.path, force=self.force, **(self.download_kwargs or {}))


@dataclass
class GoogleSource(EnsuredSource):
    """A mixin source for remote sources that must be downloaded before opening."""

    google_file_id: str
    force: bool = False

    def ensure(self) -> None:
        """Ensure the remote dataset is downloaded."""
        download_from_google(self.google_file_id, self.path, force=self.force)


@dataclass
class SimpleSource(Source):
    """A source that is directly opened.

    Supports regular files and zipped files via :func:`pystow.utils.safe_open`.
    """

    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Open the wrapped path with :func:`pystow.utils.safe_open`."""
        with safe_open(self.path) as file:
            yield file


@dataclass
class RemoteSimpleSource(RemoteSource, SimpleSource):
    """A simple source that is remote."""


@dataclass
class ArchivedSource(Source):
    """A source for a file inside an archive."""

    archive_type: ArchiveType
    inner_path: str | PurePath

    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Open the file from within a zip or tar archive."""
        with open_archive(
            self.path,
            self.inner_path,
            archive_type=self.archive_type,
            representation="text",
        ) as file:
            yield file


@dataclass
class RemoteArchivedSource(RemoteSource, ArchivedSource):
    """An archived source that is remote."""


@dataclass
class GoogleArchivedSource(GoogleSource, ArchivedSource):
    """An archived source that is remote on Google."""
