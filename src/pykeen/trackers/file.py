"""Tracking results in local files."""

import logging
import pathlib
from typing import Any, Mapping, Optional, TextIO, Union

from .base import ResultTracker

__all__ = [
    "CSVResultTracker",
]

logger = logging.getLogger(__name__)


class CSVResultTracker(ResultTracker):
    """Tracking results to a CSV file."""

    file: TextIO

    def __init__(self, path: Union[None, str, pathlib.Path] = None):
        if path is None:
            path = "results.txt"
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.file = self.path.open(mode="w")

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        logger.info(f"Logging results for {run_name} to {self.path.as_uri()}.")
        # write header
        self.file.writelines(["type,step,key,value\n"])

    def end_run(self) -> None:  # noqa: D102
        self.file.close()

    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        lines = []
        for key, value in params.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            lines.append(",".join(("parameter", 0, key, value)) + "\n")
        self.file.writelines(lines)
        self.file.flush()

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        lines = []
        for key, value in metrics.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            lines.append(",".join(("metric", step, key, value)) + "\n")
        self.file.writelines(lines)
        self.file.flush()
