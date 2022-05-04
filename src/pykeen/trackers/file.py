# -*- coding: utf-8 -*-

"""Tracking results in local files."""

import csv
import datetime
import json
import logging
import pathlib
from typing import Any, ClassVar, Mapping, Optional, TextIO, Union

from .base import ResultTracker
from ..constants import PYKEEN_LOGS
from ..utils import flatten_dictionary, normalize_path

__all__ = [
    "FileResultTracker",
    "CSVResultTracker",
    "JSONResultTracker",
]

logger = logging.getLogger(__name__)


def _format_key(key: str, prefix: Optional[str] = None) -> str:
    """Prepend prefix is necessary."""
    if prefix is None:
        return key
    return f"{prefix}.{key}"


class FileResultTracker(ResultTracker):
    """Tracking results to a file.

    Also allows monitoring experiments, e.g. by

    .. code-block::

        tail -f results.txt | grep "hits_at_10"
    """

    #: The file extension for this writer (do not include dot)
    extension: ClassVar[str]

    #: The file where the results are written to.
    file: TextIO

    def __init__(
        self,
        path: Union[None, str, pathlib.Path] = None,
        name: Optional[str] = None,
    ):
        """Initialize the tracker.

        :param path:
            The path of the log file.
        :param name: The default file name for a file if no path is given. If no default is given,
            the current time is used.
        """
        if name is None:
            name = datetime.datetime.now().isoformat()
        path = normalize_path(path, default=PYKEEN_LOGS.joinpath(f"{name}.{self.extension}"), mkdir=True, is_file=True)
        logger.info(f"Logging to {path.as_uri()}.")
        self.file = path.open(mode="w", newline="", encoding="utf8")

    # docstr-coverage: inherited
    def end_run(self, success: bool = True) -> None:  # noqa: D102
        self.file.close()


class CSVResultTracker(FileResultTracker):
    """Tracking results to a CSV file.

    Also allows monitoring experiments, e.g. by

    .. code-block::

        tail -f results.txt | grep "hits_at_10"
    """

    extension = "csv"

    #: The column names
    HEADER = "type", "step", "key", "value"

    def __init__(
        self,
        path: Union[None, str, pathlib.Path] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the tracker.

        :param path:
            The path of the log file.
        :param name: The default file name for a file if no path is given. If no default is given,
            the current time is used.
        :param kwargs:
            Additional keyword based arguments forwarded to csv.writer.
        """
        super().__init__(path=path, name=name)
        self.csv_writer = csv.writer(self.file, **kwargs)

    # docstr-coverage: inherited
    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.csv_writer.writerow(self.HEADER)

    # docstr-coverage: inherited
    def _write(
        self,
        dictionary: Mapping[str, Any],
        label: str,
        step: Optional[int],
        prefix: Optional[str],
    ) -> None:  # noqa: D102
        dictionary = flatten_dictionary(dictionary=dictionary, prefix=prefix)
        self.csv_writer.writerows((label, step, key, value) for key, value in dictionary.items())
        self.file.flush()

    # docstr-coverage: inherited
    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        self._write(dictionary=params, label="parameter", step=0, prefix=prefix)

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        self._write(dictionary=metrics, label="metric", step=step, prefix=prefix)


class JSONResultTracker(FileResultTracker):
    """Tracking results to a JSON lines file.

    Also allows monitoring experiments, e.g. by

    .. code-block::

        tail -f results.txt | grep "hits_at_10"
    """

    extension = "jsonl"

    def _write(self, obj) -> None:
        print(json.dumps(obj), file=self.file, flush=True)  # noqa:T201

    # docstr-coverage: inherited
    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        self._write({"params": params, "prefix": prefix})

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        self._write({"metrics": metrics, "prefix": prefix, "step": step})
