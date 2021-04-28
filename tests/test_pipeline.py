# -*- coding: utf-8 -*-

"""Test the PyKEEN pipeline function."""

import tempfile
import unittest

import pandas as pd
import torch

import pykeen.regularizers
from pykeen.datasets import EagerDataset, Nations
from pykeen.models import ERModel, Model
from pykeen.models.predict import (
    get_all_prediction_df, get_head_prediction_df, get_relation_prediction_df,
    get_tail_prediction_df,
)
from pykeen.models.resolve import DimensionError, make_model, make_model_cls
from pykeen.nn.modules import TransEInteraction
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.regularizers import NoRegularizer
from pykeen.training import SLCWATrainingLoop
from pykeen.triples.generation import generate_triples_factory
from pykeen.utils import resolve_device
from tests.mocks import MockModel


class TestPipeline(unittest.TestCase):
    """Test the pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up a shared result."""
        cls.device = resolve_device('cuda')
        cls.dataset = Nations()
        cls.result = pipeline(
            model='TransE',
            dataset=cls.dataset,
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
            device=cls.device,
            random_seed=42,
        )
        cls.model = cls.result.model
        cls.testing_mapped_triples = cls.dataset.testing.mapped_triples.to(cls.model.device)

    def test_predict_tails_no_novelties(self):
        """Test scoring tails without labeling as novel w.r.t. training and testing."""
        tails_df = get_tail_prediction_df(
            self.model,
            'brazil', 'intergovorgs', testing=self.testing_mapped_triples,
            triples_factory=self.dataset.training,
            add_novelties=False,
        )
        self.assertEqual(['tail_id', 'tail_label', 'score'], list(tails_df.columns))
        self.assertEqual(len(self.dataset.training.entity_to_id), len(tails_df.index))

    def test_predict_tails_remove_known(self):
        """Test scoring tails while removing non-novel triples w.r.t. training and testing."""
        tails_df = get_tail_prediction_df(
            self.model,
            'brazil', 'intergovorgs', testing=self.testing_mapped_triples,
            remove_known=True,
            triples_factory=self.dataset.training,
        )
        self.assertEqual(['tail_id', 'tail_label', 'score'], list(tails_df.columns))
        self.assertEqual({'jordan', 'brazil', 'ussr', 'burma', 'china'}, set(tails_df['tail_label']))

    def test_predict_tails_with_novelties(self):
        """Test scoring tails with labeling as novel w.r.t. training and testing."""
        tails_df = get_tail_prediction_df(
            self.model, 'brazil', 'intergovorgs',
            triples_factory=self.dataset.training,
            testing=self.testing_mapped_triples,
        )
        self.assertEqual(['tail_id', 'tail_label', 'score', 'in_training', 'in_testing'], list(tails_df.columns))
        self.assertEqual(self.model.num_entities, len(tails_df.index))
        training_tails = set(tails_df.loc[tails_df['in_training'], 'tail_label'])
        self.assertEqual({'usa', 'uk', 'netherlands', 'egypt', 'india', 'israel', 'indonesia'}, training_tails)
        testing_tails = set(tails_df.loc[tails_df['in_testing'], 'tail_label'])
        self.assertEqual({'poland', 'cuba'}, testing_tails)

    def test_predict_relations_with_novelties(self):
        """Test scoring relations with labeling as novel w.r.t. training and testing."""
        rel_df = get_relation_prediction_df(
            self.model, 'brazil', 'uk',
            triples_factory=self.dataset.training,
            testing=self.testing_mapped_triples,
        )
        self.assertEqual(['relation_id', 'relation_label', 'score', 'in_training', 'in_testing'], list(rel_df.columns))
        self.assertEqual(self.model.num_relations, len(rel_df.index))
        training_rels = set(rel_df.loc[rel_df['in_training'], 'relation_label'])
        self.assertEqual(
            {
                'weightedunvote', 'relexports', 'intergovorgs', 'timesinceally', 'exports3', 'booktranslations',
                'relbooktranslations', 'reldiplomacy', 'ngoorgs3', 'ngo', 'relngo', 'reltreaties', 'independence',
                'intergovorgs3', 'unweightedunvote', 'commonbloc2', 'relintergovorgs',
            },
            training_rels,
        )
        testing_heads = set(rel_df.loc[rel_df['in_testing'], 'relation_label'])
        self.assertEqual({'embassy'}, testing_heads)

    def test_predict_heads_with_novelties(self):
        """Test scoring heads with labeling as novel w.r.t. training and testing."""
        heads_df = get_head_prediction_df(
            self.model,
            'conferences', 'brazil',
            triples_factory=self.dataset.training,
            testing=self.testing_mapped_triples,
        )
        self.assertEqual(['head_id', 'head_label', 'score', 'in_training', 'in_testing'], list(heads_df.columns))
        self.assertEqual(self.model.num_entities, len(heads_df.index))
        training_heads = set(heads_df.loc[heads_df['in_training'], 'head_label'])
        self.assertEqual({'usa', 'india', 'ussr', 'poland', 'cuba'}, training_heads)
        testing_heads = set(heads_df.loc[heads_df['in_testing'], 'head_label'])
        self.assertEqual(set(), testing_heads)

    def test_predict_all_no_novelties(self):
        """Test scoring all triples without labeling as novel w.r.t. training and testing."""
        all_df = get_all_prediction_df(
            model=self.model,
            triples_factory=self.dataset.training,
            testing=self.testing_mapped_triples,
            add_novelties=False,
        )
        self.assertIsInstance(all_df, pd.DataFrame)
        self.assertEqual(
            ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score'],
            list(all_df.columns),
        )
        possible = self.dataset.training.num_relations * self.model.num_entities ** 2
        self.assertEqual(possible, len(all_df.index))

    def test_predict_all_remove_known(self):
        """Test scoring all triples while removing non-novel triples w.r.t. training and testing."""
        all_df = get_all_prediction_df(
            model=self.model,
            triples_factory=self.dataset.training,
            testing=self.testing_mapped_triples,
            remove_known=True,
        )
        self.assertIsInstance(all_df, pd.DataFrame)
        self.assertEqual(
            ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score'],
            list(all_df.columns),
        )
        possible = self.dataset.training.num_relations * self.model.num_entities ** 2
        known = self.dataset.training.num_triples + self.testing_mapped_triples.shape[0]
        self.assertNotEqual(possible, known, msg='testing and training triples cover all possible triples')
        self.assertEqual(possible - known, len(all_df.index))

    def test_predict_all_with_novelties(self):
        """Test scoring all triples with labeling as novel w.r.t. training and testing."""
        all_df = get_all_prediction_df(
            model=self.model,
            triples_factory=self.dataset.training,
            testing=self.testing_mapped_triples,
        )
        self.assertIsInstance(all_df, pd.DataFrame)
        self.assertEqual(
            [
                'head_id', 'head_label', 'relation_id', 'relation_label',
                'tail_id', 'tail_label', 'score', 'in_training', 'in_testing',
            ],
            list(all_df.columns),
        )
        possible = self.dataset.training.num_relations * self.model.num_entities ** 2
        self.assertEqual(possible, len(all_df.index))
        self.assertEqual(self.dataset.training.num_triples, all_df['in_training'].sum())
        self.assertEqual(self.testing_mapped_triples.shape[0], all_df['in_testing'].sum())


class TestPipelineTriples(unittest.TestCase):
    """Test applying the pipeline to triples factories."""

    def setUp(self) -> None:
        """Prepare the training, testing, and validation triples factories."""
        self.base_tf = generate_triples_factory(
            num_entities=50,
            num_relations=9,
            num_triples=500,
        )
        self.training, self.testing, self.validation = self.base_tf.split([0.8, 0.1, 0.1])

    def test_unlabeled_triples(self):
        """Test running the pipeline on unlabeled triples factories."""
        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model='TransE',
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
        )

    def test_eager_unlabeled_dataset(self):
        """Test running the pipeline on unlabeled triples factories in a dataset."""
        dataset = EagerDataset(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
        )
        _ = pipeline(
            dataset=dataset,
            model='TransE',
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
        )

    def test_interaction_instance_missing_dimensions(self):
        """Test when a dimension is missing."""
        with self.assertRaises(DimensionError) as exc:
            make_model_cls(
                dimensions={},  # missing "d"
                interaction=TransEInteraction(p=2),
            )
        self.assertIsInstance(exc.exception, DimensionError)
        self.assertEqual({'d'}, exc.exception.expected)
        self.assertEqual(set(), exc.exception.given)
        self.assertEqual("Expected dimensions dictionary with keys {'d'} but got keys set()", str(exc.exception))

    def test_interaction_instance_builder(self):
        """Test resolving an interaction model instance."""
        model = make_model(
            dimensions={"d": 3},
            interaction=TransEInteraction,
            interaction_kwargs=dict(p=2),
            triples_factory=self.training,
        )
        self.assertIsInstance(model, ERModel)
        self.assertIsInstance(model.interaction, TransEInteraction)
        self.assertEqual(2, model.interaction.p)
        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model=model,
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
            random_seed=0,
        )

    def test_interaction_builder(self):
        """Test resolving an interaction model."""
        model_cls = make_model_cls({"d": 3}, TransEInteraction(p=2))
        self._help_test_interaction_resolver(model_cls)

    def test_interaction_resolver_cls(self):
        """Test resolving the interaction function."""
        model_cls = make_model_cls({"d": 3}, TransEInteraction, {'p': 2})
        self._help_test_interaction_resolver(model_cls)

    def test_interaction_resolver_lookup(self):
        """Test resolving the interaction function."""
        model_cls = make_model_cls({"d": 3}, 'TransE', {'p': 2})
        self._help_test_interaction_resolver(model_cls)

    def _help_test_interaction_resolver(self, model_cls):
        self.assertTrue(issubclass(model_cls, ERModel))
        self.assertIsInstance(model_cls._interaction, TransEInteraction)
        self.assertEqual(2, model_cls._interaction.p)
        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model=model_cls,
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
            random_seed=0,
        )

    def test_custom_training_loop(self):
        """Test providing a custom training loop."""
        losses = []

        class ModifiedTrainingLoop(SLCWATrainingLoop):
            """A wrapper around SLCWA training loop which remembers batch losses."""

            def _forward_pass(self, *args, **kwargs):  # noqa: D102
                loss = super()._forward_pass(*args, **kwargs)
                losses.append(loss)
                return loss

        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            training_loop=ModifiedTrainingLoop,
            model='TransE',
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
            random_seed=0,
        )

        # empty lists are falsy
        self.assertTrue(losses)


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
        # As the resumption capability currently is a function of the training loop, more thorough tests can be found
        # in the test_training.py unit tests. In the tests below the handling of training loop checkpoints by the
        # pipeline is checked.

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
                checkpoint_directory=self.temporary_directory.name,
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
                checkpoint_directory=self.temporary_directory.name,
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


class TestPipelineEvaluationFiltering(unittest.TestCase):
    """Test filtering of triples during evaluation using the pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up a shared result."""
        cls.device = resolve_device('cuda')
        cls.dataset = Nations()

        cls.model = MockModel(triples_factory=cls.dataset.training)

        # The MockModel gives the highest score to the highest entity id
        max_score = cls.dataset.num_entities - 1

        # The test triples are created to yield the third highest score on both head and tail prediction
        cls.dataset.testing.mapped_triples = torch.tensor([[max_score - 2, 0, max_score - 2]])

        # Write new mapped triples to the model, since the model's triples will be used to filter
        # These triples are created to yield the highest score on both head and tail prediction for the
        # test triple at hand
        cls.dataset.training.mapped_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score],
                [max_score, 0, max_score - 2],
            ],
        )

        # The validation triples are created to yield the second highest score on both head and tail prediction for the
        # test triple at hand
        cls.dataset.validation.mapped_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score - 1],
                [max_score - 1, 0, max_score - 2],
            ],
        )

    def test_pipeline_evaluation_filtering_without_validation_triples(self):
        """Test if the evaluator's triple filtering works as expected using the pipeline."""
        results = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop_kwargs=dict(automatic_memory_optimization=False),
            training_kwargs=dict(num_epochs=0, use_tqdm=False),
            evaluator_kwargs=dict(filtered=True, automatic_memory_optimization=False),
            evaluation_kwargs=dict(use_tqdm=False),
            device=self.device,
            random_seed=42,
            filter_validation_when_testing=False,
        )
        assert results.metric_results.arithmetic_mean_rank['both']['realistic'] == 2, 'The rank should equal 2'

    def test_pipeline_evaluation_filtering_with_validation_triples(self):
        """Test if the evaluator's triple filtering with validation triples works as expected using the pipeline."""
        results = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop_kwargs=dict(automatic_memory_optimization=False),
            training_kwargs=dict(num_epochs=0, use_tqdm=False),
            evaluator_kwargs=dict(filtered=True, automatic_memory_optimization=False),
            evaluation_kwargs=dict(use_tqdm=False),
            device=self.device,
            random_seed=42,
            filter_validation_when_testing=True,
        )
        assert results.metric_results.arithmetic_mean_rank['both']['realistic'] == 1, 'The rank should equal 1'
