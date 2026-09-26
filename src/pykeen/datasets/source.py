"""A generic source wrapper."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from pystow.utils import download, open_tarfile, open_zipfile, safe_open
from pystow.utils.download import DownloadKwargs


@dataclass
class Source(ABC):
    """A configuration for a data resource."""

    #: The local path where the resource is stored on disk.
    path: Path

    @abstractmethod
    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Open the source as a file object."""


@dataclass
class RemoteSource(Source):
    """A mixin source for remote sources that must be downloaded before opening."""

    #: The remove location of the file.
    url: str
    download_kwargs: DownloadKwargs | None = None

    def ensure(self) -> None:
        """Ensure the remote dataset is downloaded."""
        download(self.url, self.path, force=False, **(self.download_kwargs or {}))

    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Download the file and open it using polymorphism."""
        self.ensure()
        with super().open() as file:
            yield file


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

    archive_type: Literal["zip", "tar"]
    inner_path: str

    @contextmanager
    def open(self) -> Generator[IO[str]]:
        """Open the file from within a zip or tar archive."""
        if self.archive_type == "zip":
            with open_zipfile(self.path, inner_path=self.inner_path) as file:
                yield file
        elif self.archive_type == "tar":
            with open_tarfile(self.path, inner_path=self.inner_path) as file:
                yield file
        else:
            raise ValueError(f"unknown {self.archive_type=}")


@dataclass
class RemoteArchivedSource(RemoteSource, ArchivedSource):
    """An archived source that is remote."""
