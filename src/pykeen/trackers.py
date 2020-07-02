# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

from typing import Any, Dict, Optional

from .utils import flatten_dictionary

__all__ = [
    'ResultTracker',
    'MLFlowResultTracker',
]


class ResultTracker:
    """A class that tracks the results from a pipeline run."""

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a run with an optional name."""

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:
        """Log parameters to result store."""

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: Optional[str] = None) -> None:
        """Log metrics to result store.

        :param metrics: The metrics to log.
        :param step: An optional step to attach the metrics to (e.g. the epoch).
        :param prefix: An optional prefix to prepend to every key in metrics.
        """

    def end_run(self) -> None:
        """End a run.

        HAS to be called after the experiment is finished.
        """


class MLFlowResultTracker(ResultTracker):
    """A tracker for MLFlow."""

    def __init__(self, tracking_uri: Optional[str] = None):
        import mlflow as _mlflow
        self.mlflow = _mlflow

        if tracking_uri is None:
            tracking_uri = 'localhost:5000'

        self.mlflow.set_tracking_uri(tracking_uri)

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.mlflow.start_run(run_name=run_name)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        self.mlflow.log_metrics(metrics=metrics, step=step)

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        self.mlflow.log_params(params=params)

    def end_run(self) -> None:  # noqa: D102
        self.mlflow.end_run()
