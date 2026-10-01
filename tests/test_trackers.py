"""Tests for result trackers."""

import json
import pathlib

import pytest

from pykeen.losses import MarginRankingLoss
from pykeen.trackers import TensorBoardResultTracker, WANDBResultTracker
from pykeen.trackers.base import ConsoleResultTracker, MultiResultTracker, PythonResultTracker
from pykeen.trackers.file import CSVResultTracker, JSONResultTracker
from tests import cases

from .utils import is_installed, needs_packages


class PythonResultTrackerTests(cases.ResultTrackerTests):
    """Tests for Python result tracker."""

    cls = PythonResultTracker


class CSVResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker


class JSONResultTrackerTests(cases.FileResultTrackerTests):
    """Tests for JSONResultTracker."""

    cls = JSONResultTracker

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


class MultiResultTrackerTests(cases.ResultTrackerTests):
    """Tests for multi tracker."""

    cls = MultiResultTracker
    kwargs = {
        "trackers": (
            ConsoleResultTracker(),
            ConsoleResultTracker(),
        )
    }


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
