# -*- coding: utf-8 -*-

"""Test the PyKEEN pipeline function."""

import unittest

from pykeen.datasets import Nations
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

        model = pipeline_result.model
        nations = Nations()
        testing_mapped_triples = nations.testing.mapped_triples.to(model.device)

        tails_df = model.predict_tails('brazil', 'intergovorgs', testing=testing_mapped_triples)
        training_tails = set(tails_df.loc[tails_df['in_training'], 'tail_label'])
        self.assertEqual({'usa', 'uk', 'netherlands', 'egypt', 'india', 'israel', 'indonesia'}, training_tails)
        testing_tails = set(tails_df.loc[tails_df['in_testing'], 'tail_label'])
        self.assertEqual({'poland', 'cuba'}, testing_tails)

        heads_df = model.predict_heads('conferences', 'brazil', testing=testing_mapped_triples)
        training_heads = set(heads_df.loc[heads_df['in_training'], 'head_label'])
        self.assertEqual({'usa', 'india', 'ussr', 'poland', 'cuba'}, training_heads)
        testing_heads = set(heads_df.loc[heads_df['in_testing'], 'head_label'])
        self.assertEqual(set(), testing_heads)

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
