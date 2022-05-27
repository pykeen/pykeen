# -*- coding: utf-8 -*-

"""Utilities and base classes for PyKEEN tracker adapters."""

import logging
import re
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Pattern, Union

from tqdm.auto import tqdm

from ..utils import flatten_dictionary

__all__ = [
    "ResultTracker",
    "ConsoleResultTracker",
    "MultiResultTracker",
    "PythonResultTracker",
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

    def end_run(self, success: bool = True) -> None:
        """End a run.

        HAS to be called after the experiment is finished.

        :param success:
            Can be used to signal failed runs. May be ignored.
        """


class PythonResultTracker(ResultTracker):
    """A tracker which stores everything in Python dictionaries.

    Example Usage: get default configuration

    .. code-block:: python

        from pykeen.pipeline import pipeline
        from pykeen.trackers import PythonResultTracker

        tracker = PythonResultTracker()
        result = pipeline(
            dataset="nations",
            model="PairRE",
            result_tracker=tracker,
        )
        print("Default configuration:")
        for k, v in tracker.configuration.items():
            print(f"{k:20} = {v}")

    """

    #: The name of the run
    run_name: Optional[str]

    #: The configuration dictionary, a mapping from name -> value
    configuration: MutableMapping[str, Any]

    #: Should metrics be stored when running ``log_metrics()``?
    store_metrics: bool

    #: The metrics, a mapping from step -> (name -> value)
    metrics: MutableMapping[Optional[int], Mapping[str, float]]

    def __init__(self, store_metrics: bool = True) -> None:
        """Initialize the tracker."""
        super().__init__()
        self.store_metrics = store_metrics
        self.configuration = dict()
        self.metrics = dict()
        self.run_name = None

    # docstr-coverage: inherited
    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.run_name = run_name

    # docstr-coverage: inherited
    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        if prefix is not None:
            params = {f"{prefix}.{key}": value for key, value in params.items()}
        self.configuration.update(params)

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        if not self.store_metrics:
            return

        if prefix is not None:
            metrics = {f"{prefix}.{key}": value for key, value in metrics.items()}
        self.metrics[step] = metrics


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
        writer: str = "tqdm",
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

        if writer == "tqdm":
            self.write = tqdm.write
        elif writer == "builtin":
            self.write = print  # noqa:T202
        elif writer == "logging":
            self.write = logging.getLogger("pykeen").info

    # docstr-coverage: inherited
    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        if run_name is not None and self.start_end_run:
            self.write(f"Starting run: {run_name}")

    # docstr-coverage: inherited
    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        if not self.track_parameters:
            return

        for key, value in flatten_dictionary(dictionary=params).items():
            if not self.parameter_filter or self.parameter_filter.match(key):
                self.write(f"Parameter: {key} = {value}")

    # docstr-coverage: inherited
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
                self.write(f"Metric: {key} = {value}")

    # docstr-coverage: inherited
    def end_run(self, success: bool = True) -> None:  # noqa: D102
        if not success:
            self.write("Run failed.")
        if self.start_end_run:
            self.write("Finished run.")


#: A hint for constructing a :class:`MultiResultTracker`
TrackerHint = Union[None, ResultTracker, Iterable[ResultTracker]]


class MultiResultTracker(ResultTracker):
    """A result tracker which delegates to multiple different result trackers."""

    trackers: List[ResultTracker]

    def __init__(self, trackers: TrackerHint = None) -> None:
        """
        Initialize the tracker.

        :param trackers:
            the base tracker(s).
        """
        if trackers is None:
            self.trackers = []
        elif isinstance(trackers, ResultTracker):
            self.trackers = [trackers]
        else:
            self.trackers = list(trackers)

    # docstr-coverage: inherited
    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        for tracker in self.trackers:
            tracker.start_run(run_name=run_name)

    # docstr-coverage: inherited
    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        for tracker in self.trackers:
            tracker.log_params(params=params, prefix=prefix)

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        for tracker in self.trackers:
            tracker.log_metrics(metrics=metrics, step=step, prefix=prefix)

    # docstr-coverage: inherited
    def end_run(self, success: bool = True) -> None:  # noqa: D102
        for tracker in self.trackers:
            tracker.end_run(success=success)

    def get_configuration(self):
        """Get the configuration from a Python result tracker."""
        tracker = next(_tracker for _tracker in self.trackers if isinstance(_tracker, PythonResultTracker))
        return tracker.configuration
