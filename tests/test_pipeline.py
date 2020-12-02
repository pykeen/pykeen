# -*- coding: utf-8 -*-

"""Test the PyKEEN pipeline function."""

import tempfile
import unittest

import pandas as pd

import pykeen.regularizers
from pykeen.datasets import Nations
from pykeen.models.base import Model
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.regularizers import NoRegularizer


class TestPipeline(unittest.TestCase):
    """Test the pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up a shared result."""
        cls.result = pipeline(
            model='TransE',
            dataset='nations',
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
        )
        cls.model = cls.result.model
        nations = Nations()
        cls.testing_mapped_triples = nations.testing.mapped_triples.to(cls.model.device)

    def test_predict_tails_no_novelties(self):
        """Test scoring tails without labeling as novel w.r.t. training and testing."""
        tails_df = self.model.predict_tails(
            'brazil', 'intergovorgs', testing=self.testing_mapped_triples,
            add_novelties=False,
        )
        self.assertEqual(['tail_id', 'tail_label', 'score'], list(tails_df.columns))
        self.assertEqual(len(self.model.triples_factory.entity_to_id), len(tails_df.index))

    def test_predict_tails_remove_known(self):
        """Test scoring tails while removing non-novel triples w.r.t. training and testing."""
        tails_df = self.model.predict_tails(
            'brazil', 'intergovorgs', testing=self.testing_mapped_triples,
            remove_known=True,
        )
        self.assertEqual(['tail_id', 'tail_label', 'score'], list(tails_df.columns))
        self.assertEqual({'jordan', 'brazil', 'ussr', 'burma', 'china'}, set(tails_df['tail_label']))

    def test_predict_tails_with_novelties(self):
        """Test scoring tails with labeling as novel w.r.t. training and testing."""
        tails_df = self.model.predict_tails('brazil', 'intergovorgs', testing=self.testing_mapped_triples)
        self.assertEqual(['tail_id', 'tail_label', 'score', 'in_training', 'in_testing'], list(tails_df.columns))
        self.assertEqual(len(self.model.triples_factory.entity_to_id), len(tails_df.index))
        training_tails = set(tails_df.loc[tails_df['in_training'], 'tail_label'])
        self.assertEqual({'usa', 'uk', 'netherlands', 'egypt', 'india', 'israel', 'indonesia'}, training_tails)
        testing_tails = set(tails_df.loc[tails_df['in_testing'], 'tail_label'])
        self.assertEqual({'poland', 'cuba'}, testing_tails)

    def test_predict_heads_with_novelties(self):
        """Test scoring heads with labeling as novel w.r.t. training and testing."""
        heads_df = self.model.predict_heads('conferences', 'brazil', testing=self.testing_mapped_triples)
        self.assertEqual(['head_id', 'head_label', 'score', 'in_training', 'in_testing'], list(heads_df.columns))
        self.assertEqual(len(self.model.triples_factory.entity_to_id), len(heads_df.index))
        training_heads = set(heads_df.loc[heads_df['in_training'], 'head_label'])
        self.assertEqual({'usa', 'india', 'ussr', 'poland', 'cuba'}, training_heads)
        testing_heads = set(heads_df.loc[heads_df['in_testing'], 'head_label'])
        self.assertEqual(set(), testing_heads)

    def test_predict_all_no_novelties(self):
        """Test scoring all triples without labeling as novel w.r.t. training and testing."""
        all_df = self.model.score_all_triples(testing=self.testing_mapped_triples, add_novelties=False)
        self.assertIsInstance(all_df, pd.DataFrame)
        self.assertEqual(
            ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score'],
            list(all_df.columns),
        )
        possible = self.model.triples_factory.num_relations * self.model.num_entities ** 2
        self.assertEqual(possible, len(all_df.index))

    def test_predict_all_remove_known(self):
        """Test scoring all triples while removing non-novel triples w.r.t. training and testing."""
        all_df = self.model.score_all_triples(testing=self.testing_mapped_triples, remove_known=True)
        self.assertIsInstance(all_df, pd.DataFrame)
        self.assertEqual(
            ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score'],
            list(all_df.columns),
        )
        possible = self.model.triples_factory.num_relations * self.model.num_entities ** 2
        known = self.model.triples_factory.num_triples + self.testing_mapped_triples.shape[0]
        self.assertNotEqual(possible, known, msg='testing and training triples cover all possible triples')
        self.assertEqual(possible - known, len(all_df.index))

    def test_predict_all_with_novelties(self):
        """Test scoring all triples with labeling as novel w.r.t. training and testing."""
        all_df = self.model.score_all_triples(testing=self.testing_mapped_triples)
        self.assertIsInstance(all_df, pd.DataFrame)
        self.assertEqual(
            [
                'head_id', 'head_label', 'relation_id', 'relation_label',
                'tail_id', 'tail_label', 'score', 'in_training', 'in_testing',
            ],
            list(all_df.columns),
        )
        possible = self.model.triples_factory.num_relations * self.model.num_entities ** 2
        self.assertEqual(possible, len(all_df.index))
        self.assertEqual(self.model.triples_factory.num_triples, all_df['in_training'].sum())
        self.assertEqual(self.testing_mapped_triples.shape[0], all_df['in_testing'].sum())


class TestPipelineCheckpoints(unittest.TestCase):
    """Test the pipeline with checkpoints."""

    def setUp(self) -> None:
        """Set up a shared result as standard to compare to."""
        self.random_seed = 123
        self.model = 'TransE'
        self.dataset = 'nations'
        self.checkpoint_name = "PyKEEN_training_loop_test_checkpoint.pt"
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Tear down the test case."""
        self.temporary_directory.cleanup()

    def test_pipeline_resumption(self):
        """Test whether the resumed LCWA pipeline creates the same results as the one shot pipeline."""
        self._test_pipeline_x_resumption(training_loop_type='LCWA')

    def test_pipeline_slcwa_resumption(self):
        """Test whether the resumed sLCWA pipeline creates the same results as the one shot pipeline."""
        self._test_pipeline_x_resumption(training_loop_type='sLCWA')

    def _test_pipeline_x_resumption(self, training_loop_type: str):
        """Test whether the resumed pipeline creates the same results as the one shot pipeline."""
        checkpoint_directory = self.temporary_directory.name

        # TODO check that there aren't any checkpoint files already existing in the right place

        result_standard = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop=training_loop_type,
            training_kwargs=dict(num_epochs=10, use_tqdm=False, use_tqdm_batch=False),
            random_seed=self.random_seed,
        )

        # Set up a shared result that runs two pipelines that should replicate the results of the standard pipeline.
        _ = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop=training_loop_type,
            training_kwargs=dict(
                num_epochs=5,
                use_tqdm=False,
                use_tqdm_batch=False,
                checkpoint_name=self.checkpoint_name,
                checkpoint_directory=checkpoint_directory,
                checkpoint_frequency=0,
            ),
            random_seed=self.random_seed,
        )

        # Resume the previous pipeline
        result_split = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop=training_loop_type,
            training_kwargs=dict(
                num_epochs=10,
                use_tqdm=False,
                use_tqdm_batch=False,
                checkpoint_name=self.checkpoint_name,
                checkpoint_directory=checkpoint_directory,
                checkpoint_frequency=0,
            ),
        )
        self.assertEqual(result_standard.losses, result_split.losses)


class TestAttributes(unittest.TestCase):
    """Test that the keywords given to the pipeline make it through."""

    def test_specify_regularizer(self):
        """Test a pipeline that uses a regularizer."""
        for regularizer, cls in [
            (None, pykeen.regularizers.NoRegularizer),
            ('no', pykeen.regularizers.NoRegularizer),
            (NoRegularizer, pykeen.regularizers.NoRegularizer),
            ('powersum', pykeen.regularizers.PowerSumRegularizer),
            ('lp', pykeen.regularizers.LpRegularizer),
        ]:
            with self.subTest(regularizer=regularizer):
                pipeline_result = pipeline(
                    model='TransE',
                    dataset='Nations',
                    regularizer=regularizer,
                    training_kwargs=dict(num_epochs=1),
                )
                self.assertIsInstance(pipeline_result, PipelineResult)
                self.assertIsInstance(pipeline_result.model, Model)
                self.assertIsInstance(pipeline_result.model.regularizer, cls)
