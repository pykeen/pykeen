# -*- coding: utf-8 -*-

"""Utilities and base classes for PyKEEN tracker adapters."""

from typing import Any, Mapping, Optional

__all__ = [
    'ResultTracker',
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
