"""
Performance tests.

These tests train models on larger datasets, and might take a long time.
"""
import pathlib
import unittest

import pytest

from pykeen.pipeline import pipeline_from_path


@pytest.mark.slow
class RGCNTests(unittest.TestCase):
    """Tests for the RGCN model."""

    def test_wn18(self):
        """Test reproducing the results for WN18."""
        result = pipeline_from_path(
            path=str(pathlib.Path(__file__).parent.parent / "src" / "pykeen" / "experiments" / "rgcn" / "schlichtkrull2018_rgcn_wn18.json")
        )
        self.assertAlmostEqual(
            first=result.metric_results.get_metric(name="hits_at_1"),
            second=0.541,
            delta=0.02,
        )
