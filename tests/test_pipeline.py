# -*- coding: utf-8 -*-

"""Test the POEM pipeline function."""

import unittest

from poem.pipeline import PipelineResult, pipeline


class TestPipeline(unittest.TestCase):
    """Test the pipeline."""

    def test_pipeline(self):
        """Test the pipeline on TransE and nations."""
        pipeline_results = pipeline(model='TransE', data_set='nations')
        self.assertIsInstance(pipeline_results, PipelineResult)
