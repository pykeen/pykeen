# -*- coding: utf-8 -*-

"""Test the PyKEEN pipeline function."""

import unittest

from pykeen.models import TransE
from pykeen.models.base import Model
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.regularizers import NoRegularizer, PowerSumRegularizer


class TestPipeline(unittest.TestCase):
    """Test the pipeline."""

    def test_pipeline(self):
        """Test the pipeline on TransE and nations."""
        pipeline_result = pipeline(
            model='TransE',
            dataset='nations',
        )
        self.assertIsInstance(pipeline_result, PipelineResult)
        self.assertIsInstance(pipeline_result.model, Model)
        self.assertIsInstance(pipeline_result.model.regularizer, NoRegularizer)

    def test_specify_regularizer(self):
        """Test a pipeline that uses a regularizer."""
        pipeline_result = pipeline(
            model=TransE,
            dataset='nations',
            regularizer='powersum',
        )
        self.assertIsInstance(pipeline_result, PipelineResult)
        self.assertIsInstance(pipeline_result.model, Model)
        self.assertIsInstance(pipeline_result.model.regularizer, PowerSumRegularizer)
