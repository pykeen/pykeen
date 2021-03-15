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
from ..utils import flatten_dictionary

__all__ = [
    'FileResultTracker',
    'CSVResultTracker',
    'JSONResultTracker',
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
        if path is None:
            if name is None:
                name = datetime.datetime.now().isoformat()
            path = PYKEEN_LOGS / f"{name}.{self.extension}"
        elif isinstance(path, str):
            path = pathlib.Path(path)
        # as_uri() requires the path to be absolute. resolve additionally also normalizes the path
        path = path.resolve()
        logger.info(f"Logging to {path.as_uri()}.")
        path.parent.mkdir(exist_ok=True, parents=True)
        self.file = path.open(mode="w", newline="", encoding="utf8")

    def end_run(self) -> None:  # noqa: D102
        self.file.close()


class CSVResultTracker(FileResultTracker):
    """Tracking results to a CSV file.

    Also allows monitoring experiments, e.g. by

    .. code-block::

        tail -f results.txt | grep "hits_at_10"
    """

    extension = 'csv'

    #: The column names
    HEADER = "type", "step", "key", "value"

    def __init__(
        self,
        path: Union[None, str, pathlib.Path] = None,
        **kwargs,
    ):
        """Initialize the tracker.

        :param path:
            The path of the log file.
        :param kwargs:
            Additional keyword based arguments forwarded to csv.writer.
        """
        super().__init__(path=path)
        self.csv_writer = csv.writer(self.file, **kwargs)

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.csv_writer.writerow(self.HEADER)

    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        self.csv_writer.writerows(
            ("parameter", 0, key, value)
            for key, value in params.items()
        )
        self.file.flush()

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        self.csv_writer.writerows(
            ("metric", step, key, value)
            for key, value in metrics.items()
        )
        self.file.flush()


class JSONResultTracker(FileResultTracker):
    """Tracking results to a JSON lines file.

    Also allows monitoring experiments, e.g. by

    .. code-block::

        tail -f results.txt | grep "hits_at_10"
    """

    extension = 'jsonl'

    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        print(json.dumps({'params': params, 'prefix': prefix}), file=self.file)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        print(json.dumps({'metrics': metrics, 'prefix': prefix, 'step': step}), file=self.file)
