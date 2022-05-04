# -*- coding: utf-8 -*-

"""An adapter for Weights and Biases."""

import os
from typing import TYPE_CHECKING, Any, Mapping, Optional

from .base import ResultTracker
from ..utils import flatten_dictionary

if TYPE_CHECKING:
    import wandb.wandb_run

__all__ = [
    "WANDBResultTracker",
]


class WANDBResultTracker(ResultTracker):
    """A tracker for Weights and Biases.

    Note that you have to perform wandb login beforehand.
    """

    #: The WANDB run
    run: "wandb.wandb_run.Run"

    def __init__(
        self,
        project: str,
        offline: bool = False,
        **kwargs,
    ):
        """Initialize result tracking via WANDB.

        :param project:
            project name your WANDB login has access to.
        :param offline:
            whether to run in offline mode, i.e, without syncing with the wandb server.
        :param kwargs:
            additional keyword arguments passed to :func:`wandb.init`.
        """
        import wandb as _wandb

        self.wandb = _wandb
        if project is None:
            raise ValueError("Weights & Biases requires a project name.")
        self.project = project

        if offline:
            os.environ[self.wandb.env.MODE] = "dryrun"
        self.kwargs = kwargs
        self.run = None

    # docstr-coverage: inherited
    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.run = self.wandb.init(project=self.project, name=run_name, **self.kwargs)

    # docstr-coverage: inherited
    def end_run(self, success: bool = True) -> None:  # noqa: D102
        self.run.finish(exit_code=0 if success else -1)
        self.run = None

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        if self.run is None:
            raise AssertionError("start_run must be called before logging any metrics")
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        self.run.log(metrics, step=step)

    # docstr-coverage: inherited
    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        if self.run is None:
            raise AssertionError("start_run must be called before logging any metrics")
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        self.run.config.update(params)
