"""Tests for result trackers."""
import pathlib
import tempfile
from typing import Any, MutableMapping

from pykeen.trackers.file import CSVResultTracker
from tests.cases import ResultTrackerTests


class CSVResultTrackerTests(ResultTrackerTests):
    """Tests for CSVResultTracker."""

    cls = CSVResultTracker

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        # prepare a temporary test directory
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp_dir.name).joinpath("test.log")
        kwargs["path"] = self.path
        return kwargs

    def tearDown(self) -> None:  # noqa: D102
        # check that file was created
        assert self.path.is_file()
        # delete intermediate files
        self.temp_dir.cleanup()
