# -*- coding: utf-8 -*-

"""An adapter for Weights and Biases."""

import os
from typing import Any, Mapping, Optional, TYPE_CHECKING

from .base import ResultTracker
from ..utils import flatten_dictionary

if TYPE_CHECKING:
    import wandb.wandb_run

__all__ = [
    'WANDBResultTracker',
]


class WANDBResultTracker(ResultTracker):
    """A tracker for Weights and Biases.

    Note that you have to perform wandb login beforehand.
    """

    #: The WANDB run
    run: 'wandb.wandb_run.Run'

    def __init__(
        self,
        project: str,
        experiment: Optional[str] = None,
        offline: bool = False,
        **kwargs,
    ):
        """Initialize result tracking via WANDB.

        :param project:
            project name your WANDB login has access to.
        :param experiment:
            The experiment name to appear on the website. If not given, WANDB will generate a random name.
        """
        import wandb as _wandb
        self.wandb = _wandb
        if project is None:
            raise ValueError('Weights & Biases requires a project name.')
        self.project = project

        if offline:
            os.environ[self.wandb.env.MODE] = 'dryrun'

        self.run = self.wandb.init(project=self.project, name=experiment, **kwargs)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        self.wandb.log(metrics, step=step)

    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        self.wandb.config.update(params)
