# -*- coding: utf-8 -*-

"""Utilities and base classes for PyKEEN tracker adapters."""

import logging
import re
from typing import Any, Mapping, Optional, Pattern, Union

from tqdm.auto import tqdm

from ..utils import flatten_dictionary

__all__ = [
    'ResultTracker',
    'ConsoleResultTracker',
]


class ResultTracker:
    """A class that tracks the results from a pipeline run."""

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a run with an optional name."""

    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:
        """Log parameters to result store."""

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log metrics to result store.

        :param metrics: The metrics to log.
        :param step: An optional step to attach the metrics to (e.g. the epoch).
        :param prefix: An optional prefix to prepend to every key in metrics.
        """

    def end_run(self) -> None:
        """End a run.

        HAS to be called after the experiment is finished.
        """


class ConsoleResultTracker(ResultTracker):
    """A class that directly prints to console."""

    def __init__(
        self,
        *,
        track_parameters: bool = True,
        parameter_filter: Union[None, str, Pattern[str]] = None,
        track_metrics: bool = True,
        metric_filter: Union[None, str, Pattern[str]] = None,
        start_end_run: bool = False,
        writer: str = 'tqdm',
    ):
        """
        Initialize the tracker.

        :param track_parameters:
            Whether to print parameters.
        :param parameter_filter:
            A regular expression to filter parameters. If None, print all parameters.
        :param track_metrics:
            Whether to print metrics.
        :param metric_filter:
            A regular expression to filter metrics. If None, print all parameters.
        :param start_end_run:
            Whether to print start/end run messages.
        :param writer:
            The writer to use - one of "tqdm", "builtin", or "logger".
        """
        self.start_end_run = start_end_run

        self.track_parameters = track_parameters
        if isinstance(parameter_filter, str):
            parameter_filter = re.compile(parameter_filter)
        self.parameter_filter = parameter_filter

        self.track_metrics = track_metrics
        if isinstance(metric_filter, str):
            metric_filter = re.compile(metric_filter)
        self.metric_filter = metric_filter

        if writer == 'tqdm':
            self.write = tqdm.write
        elif writer == 'builtin':
            self.write = print
        elif writer == 'logging':
            self.write = logging.getLogger('pykeen').info

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        if run_name is not None and self.start_end_run:
            self.write(f"Starting run: {run_name}")

    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        if not self.track_parameters:
            return

        for key, value in flatten_dictionary(dictionary=params).items():
            if not self.parameter_filter or self.parameter_filter.match(key):
                self.write(f"Parameter: {key} = {value}")

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        if not self.track_metrics:
            return

        self.write(f"Step: {step}")
        for key, value in flatten_dictionary(dictionary=metrics, prefix=prefix).items():
            if not self.metric_filter or self.metric_filter.match(key):
                self.write(f"Parameter: {key} = {value}")

    def end_run(self) -> None:  # noqa: D102
        if self.start_end_run:
            self.write("Finished run.")
