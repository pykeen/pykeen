# -*- coding: utf-8 -*-

"""An adapter for Neptune.ai."""

from typing import TYPE_CHECKING, Any, Collection, Mapping, Optional

from .base import ResultTracker
from ..utils import flatten_dictionary

if TYPE_CHECKING:
    import neptune  # noqa
    import neptune.experiments  # noqa

__all__ = [
    "NeptuneResultTracker",
]


class NeptuneResultTracker(ResultTracker):
    """A tracker for Neptune.ai."""

    project: "neptune.Project"
    session: "neptune.Session"
    experiment: "neptune.experiments.Experiment"

    def __init__(
        self,
        project_qualified_name: Optional[str] = None,
        api_token: Optional[str] = None,
        offline: bool = False,
        experiment_id: Optional[int] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Collection[str]] = None,
    ):
        """Initialize the Neptune result tracker.

        :param project_qualified_name:
            Qualified name of a project in a form of ``namespace/project_name``.
            If ``None``, the value of ``NEPTUNE_PROJECT`` environment variable will be taken. For testing,
            should be `<your username>/sandbox`
        :param api_token:
            User's API token. If ``None``, the value of ``NEPTUNE_API_TOKEN`` environment variable will be taken.

            .. note::

                It is strongly recommended to use ``NEPTUNE_API_TOKEN`` environment variable rather than
                placing your API token in plain text in your source code.
        :param offline:
            Run neptune in offline mode (uses :class:`neptune.OfflineBackend` as the backend)
        :param experiment_id:
            The identifier of a pre-existing experiment to use. If not given, will rely
            on the ``experiment_name``.
        :param experiment_name:
            The name of the experiment. If no ``experiment_id`` is given, one will be created based
            on the name.
        :param tags: A collection of tags to add to the experiment
        """
        import neptune

        if offline:
            self.session = neptune.Session(backend=neptune.OfflineBackend())
        else:
            self.session = neptune.Session.with_default_backend(api_token=api_token)

        self.project = self.session.get_project(project_qualified_name)

        if experiment_id is None and experiment_name is None:
            raise ValueError("need experiment_name if no experiment_id is given")
        if experiment_id is None:
            self.experiment = self.project.create_experiment(name=experiment_name)
        else:
            self.experiment = self.project.get_experiments(id=experiment_id)[0]

        if tags:
            self.experiment.append_tags(*tags)

    # docstr-coverage: inherited
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        metrics = flatten_dictionary(metrics, prefix=prefix)
        for k, v in metrics.items():
            self._help_log(k, step, v)

    # docstr-coverage: inherited
    def log_params(self, params: Mapping[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        params = flatten_dictionary(params, prefix=prefix)
        for k, v in params.items():
            self._help_log(k, v)

    def _help_log(self, k, x, y=None):
        if (y is None and isinstance(x, float)) or (y is not None and isinstance(y, float)):
            self.experiment.log_metric(k, x)
        else:
            self.experiment.log_text(k, str(x))
