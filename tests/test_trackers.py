"""Tests for result trackers."""

import csv
import io
import json
import pathlib
from collections.abc import Mapping, MutableMapping
from typing import Any

import pytest

from pykeen.losses import MarginRankingLoss
from pykeen.trackers import TensorBoardResultTracker, WANDBResultTracker
from pykeen.trackers.base import ConsoleResultTracker, MultiResultTracker, PythonResultTracker
from pykeen.trackers.file import CSVResultTracker, JSONResultTracker
from tests import cases

from .utils import is_installed, needs_packages


def _verify_python_log_metrics(
    tracker: PythonResultTracker, metrics: Mapping[str, Any], step: int, prefix: str | None
) -> None:
    # note: the Python result tracker only prefixes the top-level keys, but does not flatten nested metrics
    if prefix is not None:
        metrics = {f"{prefix}.{key}": value for key, value in metrics.items()}
    assert tracker.metrics[step] == metrics


class PythonResultTrackerTests(cases.ResultTrackerTests):
    """Tests for Python result tracker."""

    cls = PythonResultTracker

    def _verify_start_run(self, run_name: str) -> None:
        assert self.instance.run_name == run_name

    def _verify_log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str | None,
        flat_metrics: Mapping[str, float],
    ) -> None:
        _verify_python_log_metrics(tracker=self.instance, metrics=metrics, step=step, prefix=prefix)

    def _verify_log_params(self, params: Mapping[str, Any], flat_params: Mapping[str, Any]) -> None:
        assert self.instance.configuration == params


class CSVResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker

    def _rows(self) -> list[list[str]]:
        return list(csv.reader(io.StringIO(self._read())))

    def _verify_start_run(self, run_name: str) -> None:
        assert self._rows() == [list(CSVResultTracker.HEADER)]

    def _verify_log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str | None,
        flat_metrics: Mapping[str, float],
    ) -> None:
        expected = [["metric", str(step), key, str(value)] for key, value in flat_metrics.items()]
        assert self._rows()[-len(expected) :] == expected

    def _verify_log_params(self, params: Mapping[str, Any], flat_params: Mapping[str, Any]) -> None:
        expected = [["parameter", "0", key, str(value)] for key, value in flat_params.items()]
        assert self._rows() == expected


class JSONResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for JSONResultTracker."""

    cls = JSONResultTracker

    def _last_record(self) -> dict[str, Any]:
        return json.loads(self._read().splitlines()[-1])

    def _verify_log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str | None,
        flat_metrics: Mapping[str, float],
    ) -> None:
        assert self._last_record() == {"metrics": metrics, "prefix": prefix, "step": step}

    def _verify_log_params(self, params: Mapping[str, Any], flat_params: Mapping[str, Any]) -> None:
        assert self._last_record() == {"params": params, "prefix": None}

    def test_log_params_not_serializable(self):
        """Test logging parameters which are not JSON serializable."""
        loss = MarginRankingLoss()
        self.instance.log_params(params={"model_kwargs": {"loss": loss}})
        self.instance.file.flush()
        (line,) = self.path.read_text().splitlines()
        assert json.loads(line)["params"] == {"model_kwargs": {"loss": str(loss)}}


class ConsoleResultTrackerTests(cases.ResultTrackerTests):
    """Tests for console tracker."""

    cls = ConsoleResultTracker

    def post_instantiation_hook(self) -> None:
        """Capture the written lines."""
        self.lines: list[str] = []
        self.instance.write = self.lines.append

    def _verify_log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str | None,
        flat_metrics: Mapping[str, float],
    ) -> None:
        expected = [f"Step: {step}", *(f"Metric: {key} = {value}" for key, value in flat_metrics.items())]
        assert self.lines[-len(expected) :] == expected

    def _verify_log_params(self, params: Mapping[str, Any], flat_params: Mapping[str, Any]) -> None:
        assert self.lines == [f"Parameter: {key} = {value}" for key, value in flat_params.items()]


class MultiResultTrackerTests(cases.ResultTrackerTests):
    """Tests for multi tracker."""

    cls = MultiResultTracker

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # use fresh trackers for each test, whose state can be inspected
        kwargs["trackers"] = (PythonResultTracker(), PythonResultTracker())
        return kwargs

    def _verify_start_run(self, run_name: str) -> None:
        for tracker in self.instance.trackers:
            assert tracker.run_name == run_name

    def _verify_log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str | None,
        flat_metrics: Mapping[str, float],
    ) -> None:
        for tracker in self.instance.trackers:
            _verify_python_log_metrics(tracker=tracker, metrics=metrics, step=step, prefix=prefix)

    def _verify_log_params(self, params: Mapping[str, Any], flat_params: Mapping[str, Any]) -> None:
        for tracker in self.instance.trackers:
            assert tracker.configuration == params


@needs_packages("tensorboard")
class TensorboardTrackerTests(cases.ResultTrackerTests):
    """Tests for TensorBoard tracker."""

    cls = TensorBoardResultTracker


@pytest.mark.skipif(not is_installed("wandb"), reason="Missing required package: wandb.")
def test_wandb_log_params_allow_val_change(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ``allow_val_change`` of the Weights & Biases tracker also applies to logging parameters."""
    import wandb

    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_MODE", "offline")

    tracker = WANDBResultTracker(project="test", allow_val_change=True)
    tracker.start_run(run_name="test")
    tracker.log_params(params={"num_epochs": 5})
    tracker.log_params(params={"num_epochs": 10})
    assert tracker.run is not None
    assert tracker.run.config["num_epochs"] == 10
    tracker.end_run()

    tracker = WANDBResultTracker(project="test")
    tracker.start_run(run_name="test")
    tracker.log_params(params={"num_epochs": 5})
    with pytest.raises(wandb.sdk.lib.config_util.ConfigError):
        tracker.log_params(params={"num_epochs": 10})
    tracker.end_run()


def test_console_log_params_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the console tracker prepends the prefix to logged parameters."""
    ConsoleResultTracker(writer="builtin").log_params(params={"loss": {"margin": 1.0}}, prefix="model")
    assert capsys.readouterr().out == "Parameter: model.loss.margin = 1.0\n"
