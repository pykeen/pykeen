# -*- coding: utf-8 -*-

"""Result trackers in PyKEEN."""

import os
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Type, Union

from .utils import flatten_dictionary, get_cls, normalize_string

__all__ = [
    'get_result_tracker_cls',
    'ResultTracker',
    'MLFlowResultTracker',
    'WANDBResultTracker',
]

if TYPE_CHECKING:
    import wandb.wandb_run


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

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_id: Optional[int] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize result tracking via MLFlow.

        :param tracking_uri:
            The tracking uri.
        :param experiment_id:
            The experiment ID. If given, this has to be the ID of an existing experiment in MFLow. Has priority over
            experiment_name.
        :param experiment_name:
            The experiment name. If this experiment name exists, add the current run to this experiment. Otherwise
            create an experiment of the given name.
        :param tags:
            The additional run details which are presented as tags to be logged
        """
        import mlflow as _mlflow
        self.mlflow = _mlflow
        self.tags = tags

        self.mlflow.set_tracking_uri(tracking_uri)
        if experiment_id is not None:
            experiment = self.mlflow.get_experiment(experiment_id=experiment_id)
            experiment_name = experiment.name
        if experiment_name is not None:
            self.mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        self.mlflow.start_run(run_name=run_name)
        if self.tags is not None:
            self.mlflow.set_tags(tags=self.tags)

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
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        self.wandb.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        self.wandb.config.update(params)


#: A mapping of trackers' names to their implementations
trackers: Mapping[str, Type[ResultTracker]] = {
    normalize_string(tracker.__name__, suffix='ResultTracker'): tracker
    for tracker in ResultTracker.__subclasses__()
}


def get_result_tracker_cls(query: Union[None, str, Type[ResultTracker]]) -> Type[ResultTracker]:
    """Get the tracker class."""
    return get_cls(
        query,
        base=ResultTracker,
        lookup_dict=trackers,
        default=ResultTracker,
    )
