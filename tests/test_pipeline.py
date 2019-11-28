# -*- coding: utf-8 -*-

"""Test the POEM pipeline function."""

import unittest

from poem.models import TransE
from poem.models.base import BaseModule
from poem.pipeline import PipelineResult, pipeline
from poem.regularizers import NoRegularizer, PowerSumRegularizer


class TestPipeline(unittest.TestCase):
    """Test the pipeline."""

    def test_pipeline(self):
        """Test the pipeline on TransE and nations."""
        pipeline_result = pipeline(
            model='TransE',
            data_set='nations',
        )
        self.assertIsInstance(pipeline_result, PipelineResult)
        self.assertIsInstance(pipeline_result.model, BaseModule)
        self.assertIsInstance(pipeline_result.model.regularizer, NoRegularizer)

    def test_specify_regularizer(self):
        """Test a pipeline that uses a regularizer."""
        pipeline_result = pipeline(
            model=TransE,
            data_set='nations',
            regularizer='powersum',
        )
        self.assertIsInstance(pipeline_result, PipelineResult)
        self.assertIsInstance(pipeline_result.model, BaseModule)
        self.assertIsInstance(pipeline_result.model.regularizer, PowerSumRegularizer)
