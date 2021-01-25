"""Tracking results in local files."""
import csv
import datetime
import logging
import pathlib
from typing import Any, Mapping, Optional, TextIO, Union

import pystow

from .base import ResultTracker

__all__ = [
    "CSVResultTracker",
]

logger = logging.getLogger(__name__)


def _format_key(key: str, prefix: Optional[str] = None) -> str:
    """Prepend prefix is necessary."""
    if prefix is None:
        return key
    return f"{prefix}.{key}"


class CSVResultTracker(ResultTracker):
    """
    Tracking results to a CSV file.

    Also allows monitoring experiments, e.g. by

    .. code ::
        tail -f results.txt | grep "hits_at_10"
    """

    #: The column names
    HEADER = ("type", "step", "key", "value")

    #: The file where the results are written to.
    file: TextIO

    def __init__(
        self,
        path: Union[None, str, pathlib.Path] = None,
        **kwargs,
    ):
        """
        Initialize the tracker.

        :param path:
            The path of the log file.
        :param kwargs:
            Additional keyword based arguments forwarded to csv.writer.
        """
        if path is None:
            path = pystow.get("logs", f"{datetime.datetime.now().isoformat()}.csv")
        logger.info(f"Logging to {path.as_uri()}.")
        path.parent.mkdir(exist_ok=True, parents=True)
        self.file = path.open("w", newline="", encoding="utf8")
        self.csv_writer = csv.writer(self.file, **kwargs)

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.csv_writer.writerow(self.HEADER)

    def end_run(self) -> None:  # noqa: D102
        self.file.close()

    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        self.csv_writer.writerows(
            ("parameter", 0, _format_key(key=key, prefix=prefix), value)
            for key, value in params.items()
        )
        self.file.flush()

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        self.csv_writer.writerows(
            ("metric", step, _format_key(key=key, prefix=prefix), value)
            for key, value in metrics.items()
        )
        self.file.flush()
