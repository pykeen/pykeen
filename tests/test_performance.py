# -*- coding: utf-8 -*-

"""Performance tests.

These tests train models on larger datasets, and might take a long time.
"""

import os
import unittest

import pytest

from pykeen.experiments.cli import HERE
from pykeen.pipeline import pipeline_from_path


@pytest.mark.slow
class RGCNTests(unittest.TestCase):
    """Tests for the RGCN model."""

    def test_wn18(self):
        """Test reproducing the results for WN18."""
        path = os.path.join(HERE, "rgcn", "schlichtkrull2018_rgcn_wn18.json")
        result = pipeline_from_path(path)
        self.assertAlmostEqual(
            first=result.metric_results.get_metric(name="hits_at_1"),
            second=0.541,
            delta=0.02,
        )
