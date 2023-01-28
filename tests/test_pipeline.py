# -*- coding: utf-8 -*-

"""Test the PyKEEN pipeline function."""

import itertools
import pathlib
import tempfile
import unittest
from typing import Type
from unittest import mock

import pytest
import torch

import pykeen.regularizers
from pykeen.datasets import EagerDataset, Nations
from pykeen.models import ERModel, FixedModel, Model
from pykeen.models.resolve import DimensionError, make_model, make_model_cls
from pykeen.nn.modules import TransEInteraction
from pykeen.nn.representation import Embedding
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.pipeline.api import replicate_pipeline_from_config
from pykeen.regularizers import NoRegularizer
from pykeen.sampling.negative_sampler import NegativeSampler
from pykeen.training import SLCWATrainingLoop
from pykeen.triples.generation import generate_triples_factory
from pykeen.triples.triples_factory import CoreTriplesFactory, TriplesFactory
from pykeen.utils import resolve_device

from .utils import needs_packages


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
            model="TransE",
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
            model="TransE",
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
        self.assertEqual({"d"}, exc.exception.expected)
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
        model_cls = make_model_cls({"d": 3}, TransEInteraction, {"p": 2})
        self._help_test_interaction_resolver(model_cls)

    def test_interaction_resolver_lookup(self):
        """Test resolving the interaction function."""
        model_cls = make_model_cls({"d": 3}, "TransE", {"p": 2})
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
            model="TransE",
            training_kwargs=dict(num_epochs=1, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
            random_seed=0,
        )

        # empty lists are falsy
        self.assertTrue(losses)

    @needs_packages("matplotlib", "seaborn")
    def test_plot(self):
        """Test plotting."""
        result = pipeline(dataset="nations", model="transe", training_kwargs=dict(num_epochs=0))
        fig, axes = result.plot()
        assert fig is not None and axes is not None

    def test_with_evaluation_loop_callback(self):
        """Smoke-Test for running pipeline with evaluation loop callback."""
        dataset = Nations()
        result = pipeline(
            dataset=dataset,
            model="mure",
            training_kwargs=dict(
                num_epochs=2,
                callbacks="evaluation-loop",
                callback_kwargs=dict(
                    frequency=1,
                    prefix="validation",
                    factory=dataset.validation,
                    additional_filter_triples=dataset.training,
                ),
            ),
        )
        assert result is not None


class TestPipelineReplicate(unittest.TestCase):
    """Test the replication with pipeline."""

    def setUp(self) -> None:  # noqa: D102
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = pathlib.Path(self.tmp_dir.name)

    def tearDown(self) -> None:  # noqa: D102
        self.tmp_dir.cleanup()

    def test_replicate_pipeline_from_config(self):
        """Test replication from config."""
        replicate_pipeline_from_config(
            config=dict(
                metadata=dict(),
                pipeline=dict(
                    dataset="nations",
                    model="transe",
                ),
                results={
                    "best": {"hits_at_k": {"10": 0.538}, "mean_rank": 163},
                },
            ),
            directory=self.tmp_dir_path,
            replicates=1,
        )


class TestPipelineCheckpoints(unittest.TestCase):
    """Test the pipeline with checkpoints."""

    def setUp(self) -> None:
        """Set up a shared result as standard to compare to."""
        self.random_seed = 123
        self.model = "TransE"
        self.dataset = "nations"
        self.checkpoint_name = "PyKEEN_training_loop_test_checkpoint.pt"
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Tear down the test case."""
        self.temporary_directory.cleanup()

    def test_pipeline_resumption(self):
        """Test whether the resumed LCWA pipeline creates the same results as the one shot pipeline."""
        self._test_pipeline_x_resumption(training_loop_type="LCWA")

    def test_pipeline_slcwa_resumption(self):
        """Test whether the resumed sLCWA pipeline creates the same results as the one shot pipeline."""
        self._test_pipeline_x_resumption(training_loop_type="sLCWA")

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
            (None, None),
            ("no", pykeen.regularizers.NoRegularizer),
            (NoRegularizer, pykeen.regularizers.NoRegularizer),
            ("powersum", pykeen.regularizers.PowerSumRegularizer),
            ("lp", pykeen.regularizers.LpRegularizer),
        ]:
            with self.subTest(regularizer=regularizer):
                pipeline_result = pipeline(
                    model="TransE",
                    dataset="Nations",
                    regularizer=regularizer,
                    training_kwargs=dict(num_epochs=1),
                )
                self.assertIsInstance(pipeline_result, PipelineResult)
                self.assertIsInstance(pipeline_result.model, Model)
                for r in itertools.chain(
                    pipeline_result.model.entity_representations, pipeline_result.model.relation_representations
                ):
                    if isinstance(r, Embedding):
                        if cls is None:
                            self.assertIsNone(r.regularizer)
                        else:
                            self.assertIsInstance(r.regularizer, cls)


class TestPipelineEvaluationFiltering(unittest.TestCase):
    """Test filtering of triples during evaluation using the pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up a shared result."""
        cls.device = resolve_device("cuda")
        cls.dataset = Nations()

        cls.model = FixedModel(triples_factory=cls.dataset.training)

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
        assert results.metric_results.get_metric("mr") == 2, "The rank should equal 2"

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
        assert results.metric_results.get_metric("mr") == 1, "The rank should equal 1"


def test_negative_sampler_kwargs():
    """Test whether negative sampler kwargs are correctly passed through."""
    # cf. https://github.com/pykeen/pykeen/issues/1118

    _num_neg_per_pos = 100

    # save a reference to the old init *before* mocking
    old_init = NegativeSampler.__init__

    def mock_init(*args, **kwargs):
        """Mock init method to check if kwarg arrives."""
        assert kwargs.get("num_negs_per_pos") == _num_neg_per_pos
        old_init(*args, **kwargs)

    # run a small pipline
    with mock.patch.object(NegativeSampler, "__init__", mock_init):
        pipeline(
            # use sampled training loop ...
            training_loop="slcwa",
            # ... without explicitly selecting a negative sampler ...
            negative_sampler=None,
            # ... but providing custom kwargs
            negative_sampler_kwargs=dict(num_negs_per_pos=_num_neg_per_pos),
            # other parameters for fast test
            dataset="nations",
            model="distmult",
            epochs=0,
        )


@pytest.mark.parametrize("tf_cls", [CoreTriplesFactory, TriplesFactory])
def test_loading_training_triples_factory(tf_cls: Type[CoreTriplesFactory]):
    """Test re-loading the training triples factory."""
    result = pipeline(model="rescal", dataset="nations", training_kwargs=dict(num_epochs=0))
    with tempfile.TemporaryDirectory() as directory:
        result.save_to_directory(directory)
        tf_cls.from_path_binary(pathlib.Path(directory, "training_triples"))
