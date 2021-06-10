# -*- coding: utf-8 -*-

"""Test cases for PyKEEN."""

import logging
import os
import pathlib
import tempfile
import timeit
import traceback
import unittest
from abc import ABC, abstractmethod
from typing import (
    Any, ClassVar, Collection, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar,
)
from unittest.case import SkipTest
from unittest.mock import patch

import pytest
import torch
import unittest_templates
from click.testing import CliRunner, Result
from torch import optim
from torch.nn import functional
from torch.optim import Adagrad, SGD

import pykeen.models
import pykeen.nn.message_passing
import pykeen.nn.weighting
from pykeen.datasets import Nations
from pykeen.datasets.base import LazyDataset
from pykeen.datasets.kinships import KINSHIPS_TRAIN_PATH
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH
from pykeen.losses import Loss, PairwiseLoss, PointwiseLoss, SetwiseLoss, UnsupportedLabelSmoothingError
from pykeen.models import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model, RESCAL
from pykeen.models.cli import build_cli_from_cls
from pykeen.nn.emb import RepresentationModule
from pykeen.nn.modules import FunctionalInteraction, Interaction, LiteralInteraction
from pykeen.optimizers import optimizer_resolver
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.trackers import ResultTracker
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop, TrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.typing import HeadRepresentation, MappedTriples, RelationRepresentation, TailRepresentation
from pykeen.utils import all_in_bounds, get_batchnorm_modules, resolve_device, set_random_seed, unpack_singletons
from tests.constants import EPSILON
from tests.mocks import CustomRepresentations
from tests.utils import rand

T = TypeVar("T")

logger = logging.getLogger(__name__)


class GenericTestCase(unittest_templates.GenericTestCase[T]):
    """Generic tests."""

    generator: torch.Generator

    def pre_setup_hook(self) -> None:
        """Instantiate a generator for usage in the test case."""
        self.generator = set_random_seed(seed=42)[1]


class DatasetTestCase(unittest.TestCase):
    """A test case for quickly defining common tests for datasets."""

    #: The expected number of entities
    exp_num_entities: ClassVar[int]
    #: The expected number of relations
    exp_num_relations: ClassVar[int]
    #: The expected number of triples
    exp_num_triples: ClassVar[int]
    #: The tolerance on expected number of triples, for randomized situations
    exp_num_triples_tolerance: ClassVar[Optional[int]] = None

    #: The dataset to test
    dataset_cls: ClassVar[Type[LazyDataset]]
    #: The instantiated dataset
    dataset: LazyDataset

    #: Should the validation be assumed to have been loaded with train/test?
    autoloaded_validation: ClassVar[bool] = False

    def test_dataset(self):
        """Generic test for datasets."""
        self.assertIsInstance(self.dataset, LazyDataset)

        # Not loaded
        self.assertIsNone(self.dataset._training)
        self.assertIsNone(self.dataset._testing)
        self.assertIsNone(self.dataset._validation)
        self.assertFalse(self.dataset._loaded)
        self.assertFalse(self.dataset._loaded_validation)

        # Load
        self.dataset._load()

        self.assertIsInstance(self.dataset.training, TriplesFactory)
        self.assertIsInstance(self.dataset.testing, TriplesFactory)
        self.assertTrue(self.dataset._loaded)

        if self.autoloaded_validation:
            self.assertTrue(self.dataset._loaded_validation)
        else:
            self.assertFalse(self.dataset._loaded_validation)
            self.dataset._load_validation()

        self.assertIsInstance(self.dataset.validation, TriplesFactory)

        self.assertIsNotNone(self.dataset._training)
        self.assertIsNotNone(self.dataset._testing)
        self.assertIsNotNone(self.dataset._validation)
        self.assertTrue(self.dataset._loaded)
        self.assertTrue(self.dataset._loaded_validation)

        self.assertEqual(self.dataset.num_entities, self.exp_num_entities)
        self.assertEqual(self.dataset.num_relations, self.exp_num_relations)

        num_triples = sum(
            triples_factory.num_triples for
            triples_factory in (self.dataset._training, self.dataset._testing, self.dataset._validation)
        )
        if self.exp_num_triples_tolerance is None:
            self.assertEqual(self.exp_num_triples, num_triples)
        else:
            self.assertAlmostEqual(self.exp_num_triples, num_triples, delta=self.exp_num_triples_tolerance)

        # Test caching
        start = timeit.default_timer()
        _ = self.dataset.training
        end = timeit.default_timer()
        # assert (end - start) < 1.0e-02
        self.assertAlmostEqual(start, end, delta=1.0e-02, msg='Caching should have made this operation fast')

        # Test consistency of training / validation / testing mapping
        training = self.dataset.training
        for part, factory in self.dataset.factory_dict.items():
            if not isinstance(factory, TriplesFactory):
                logger.warning("Skipping mapping consistency checks since triples factory does not provide mappings.")
                continue
            if part == "training":
                continue
            assert training.entity_to_id == factory.entity_to_id
            assert training.num_entities == factory.num_entities
            assert training.relation_to_id == factory.relation_to_id
            assert training.num_relations == factory.num_relations


class LocalDatasetTestCase(DatasetTestCase):
    """A test case for datasets that don't need a cache directory."""

    def setUp(self):
        """Set up the test case."""
        self.dataset = self.dataset_cls()


class CachedDatasetCase(DatasetTestCase):
    """A test case for datasets that need a cache directory."""

    #: The directory, if there is caching
    directory: Optional[tempfile.TemporaryDirectory]

    def setUp(self):
        """Set up the test with a temporary cache directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.dataset = self.dataset_cls(cache_root=self.directory.name)

    def tearDown(self) -> None:
        """Tear down the test case by cleaning up the temporary cache directory."""
        self.directory.cleanup()


class LossTestCase(GenericTestCase[Loss]):
    """Base unittest for loss functions."""

    #: The batch size
    batch_size: ClassVar[int] = 3

    #: The number of negatives per positive for sLCWA training loop.
    num_neg_per_pos: ClassVar[int] = 7

    #: The number of entities LCWA training loop / label smoothing.
    num_entities: ClassVar[int] = 7

    def _check_loss_value(self, loss_value: torch.FloatTensor) -> None:
        """Check loss value dimensionality, and ability for backward."""
        # test reduction
        self.assertEqual(0, loss_value.ndim)

        # test finite loss value
        self.assertTrue(torch.isfinite(loss_value))

        # Test backward
        loss_value.backward()

    def help_test_process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        batch_filter: Optional[torch.BoolTensor] = None,
    ):
        """Help test processing scores from SLCWA training loop."""
        loss_value = self.instance.process_slcwa_scores(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            label_smoothing=None,
            batch_filter=batch_filter,
            num_entities=self.num_entities,
        )
        self._check_loss_value(loss_value=loss_value)

    def test_process_slcwa_scores(self):
        """Test processing scores from SLCWA training loop."""
        positive_scores = torch.rand(self.batch_size, 1, requires_grad=True)
        negative_scores = torch.rand(self.batch_size, self.num_neg_per_pos, requires_grad=True)
        self.help_test_process_slcwa_scores(positive_scores=positive_scores, negative_scores=negative_scores)

    def test_process_slcwa_scores_filtered(self):
        """Test processing scores from SLCWA training loop with filtering."""
        positive_scores = torch.rand(self.batch_size, 1, requires_grad=True)
        negative_scores = torch.rand(self.batch_size, self.num_neg_per_pos, requires_grad=True)
        batch_filter = torch.rand(self.batch_size, self.num_neg_per_pos) < 0.5
        self.help_test_process_slcwa_scores(
            positive_scores=positive_scores,
            negative_scores=negative_scores[batch_filter],
            batch_filter=batch_filter,
        )

    def test_process_lcwa_scores(self):
        """Test processing scores from LCWA training loop without smoothing."""
        self.help_test_process_lcwa_scores(label_smoothing=None)

    def test_process_lcwa_scores_smooth(self):
        """Test processing scores from LCWA training loop with smoothing."""
        try:
            self.help_test_process_lcwa_scores(label_smoothing=0.01)
        except UnsupportedLabelSmoothingError as error:
            raise SkipTest from error

    def help_test_process_lcwa_scores(self, label_smoothing):
        """Help test processing scores from LCWA training loop."""
        predictions = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        labels = (torch.rand(self.batch_size, self.num_entities, requires_grad=True) > 0.8).float()
        loss_value = self.instance.process_lcwa_scores(
            predictions=predictions,
            labels=labels,
            label_smoothing=label_smoothing,
            num_entities=self.num_entities,
        )
        self._check_loss_value(loss_value=loss_value)

    def test_optimization_direction_lcwa(self):
        """Test whether the loss leads to increasing positive scores, and decreasing negative scores."""
        labels = torch.as_tensor(data=[0, 1], dtype=torch.get_default_dtype()).view(1, -1)
        predictions = torch.zeros(1, 2, requires_grad=True)
        optimizer = optimizer_resolver.make(query=None, params=[predictions])
        for _ in range(10):
            optimizer.zero_grad()
            loss = self.instance.process_lcwa_scores(predictions=predictions, labels=labels)
            loss.backward()
            optimizer.step()

        # negative scores decreased compared to positive ones
        assert predictions[0, 0] < predictions[0, 1] - 1.0e-06

    def test_optimization_direction_slcwa(self):
        """Test whether the loss leads to increasing positive scores, and decreasing negative scores."""
        positive_scores = torch.zeros(self.batch_size, requires_grad=True)
        negative_scores = torch.zeros(self.batch_size, self.num_neg_per_pos, requires_grad=True)
        optimizer = optimizer_resolver.make(query=None, params=[positive_scores, negative_scores])
        for _ in range(10):
            optimizer.zero_grad()
            loss = self.instance.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
            )
            loss.backward()
            optimizer.step()

        # negative scores decreased compared to positive ones
        assert (negative_scores < positive_scores.unsqueeze(dim=1) - 1.0e-06).all()


class PointwiseLossTestCase(LossTestCase):
    """Base unit test for label-based losses."""

    #: The number of entities.
    num_entities: int = 17

    def test_type(self):
        """Test the loss is the right type."""
        self.assertIsInstance(self.instance, PointwiseLoss)

    def test_label_loss(self):
        """Test ``forward(logits, labels)``."""
        logits = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        labels = functional.normalize(torch.rand(self.batch_size, self.num_entities, requires_grad=False), p=1, dim=-1)
        loss_value = self.instance(
            logits,
            labels,
        )
        self._check_loss_value(loss_value)


class PairwiseLossTestCase(LossTestCase):
    """Base unit test for pair-wise losses."""

    #: The number of negative samples
    num_negatives: int = 5

    def test_type(self):
        """Test the loss is the right type."""
        self.assertIsInstance(self.instance, PairwiseLoss)

    def test_pair_loss(self):
        """Test ``forward(pos_scores, neg_scores)``."""
        pos_scores = torch.rand(self.batch_size, 1, requires_grad=True)
        neg_scores = torch.rand(self.batch_size, self.num_negatives, requires_grad=True)
        loss_value = self.instance(
            pos_scores,
            neg_scores,
        )
        self._check_loss_value(loss_value)


class SetwiseLossTestCase(LossTestCase):
    """Unit tests for setwise losses."""

    #: The number of entities.
    num_entities: int = 13

    def test_type(self):
        """Test the loss is the right type."""
        self.assertIsInstance(self.instance, SetwiseLoss)


class InteractionTestCase(
    GenericTestCase[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
    ABC,
):
    """Generic test for interaction functions."""

    dim: int = 2
    batch_size: int = 3
    num_relations: int = 5
    num_entities: int = 7

    shape_kwargs = dict()

    def post_instantiation_hook(self) -> None:
        """Initialize parameters."""
        self.instance.reset_parameters()

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
    ):
        shape_kwargs = dict(self.shape_kwargs)
        shape_kwargs.setdefault("d", self.dim)
        result = tuple(
            tuple(
                torch.rand(*prefix_shape, *(shape_kwargs[dim] for dim in weight_shape), requires_grad=True)
                for weight_shape in weight_shapes
            )
            for prefix_shape, weight_shapes in zip(
                shapes,
                [self.instance.entity_shape, self.instance.relation_shape, self.instance.entity_shape],
            )
        )
        return unpack_singletons(*result)

    def _check_scores(self, scores: torch.FloatTensor, exp_shape: Tuple[int, ...]):
        """Check shape, dtype and gradients of scores."""
        assert torch.is_tensor(scores)
        assert scores.dtype == torch.float32
        assert scores.ndimension() == len(exp_shape)
        assert scores.shape == exp_shape
        assert scores.requires_grad
        self._additional_score_checks(scores)

    def _additional_score_checks(self, scores):
        """Additional checks for scores."""

    @property
    def _score_batch_sizes(self) -> Iterable[int]:
        """Return the list of batch sizes to test."""
        if get_batchnorm_modules(self.instance):
            return [self.batch_size]
        return [1, self.batch_size]

    def test_score_hrt(self):
        """Test score_hrt."""
        for batch_size in self._score_batch_sizes:
            h, r, t = self._get_hrt(
                (batch_size,),
                (batch_size,),
                (batch_size,),
            )
            scores = self.instance.score_hrt(h=h, r=r, t=t)
            self._check_scores(scores=scores, exp_shape=(batch_size, 1))

    def test_score_h(self):
        """Test score_h."""
        for batch_size in self._score_batch_sizes:
            h, r, t = self._get_hrt(
                (self.num_entities,),
                (batch_size,),
                (batch_size,),
            )
            scores = self.instance.score_h(all_entities=h, r=r, t=t)
            self._check_scores(scores=scores, exp_shape=(batch_size, self.num_entities))

    def test_score_h_slicing(self):
        """Test score_h with slicing."""
        #: The equivalence for models with batch norm only holds in evaluation mode
        self.instance.eval()
        h, r, t = self._get_hrt(
            (self.num_entities,),
            (self.batch_size,),
            (self.batch_size,),
        )
        scores = self.instance.score_h(all_entities=h, r=r, t=t, slice_size=self.num_entities // 2 + 1)
        scores_no_slice = self.instance.score_h(all_entities=h, r=r, t=t, slice_size=None)
        self._check_close_scores(scores=scores, scores_no_slice=scores_no_slice)

    def test_score_r(self):
        """Test score_r."""
        for batch_size in self._score_batch_sizes:
            h, r, t = self._get_hrt(
                (batch_size,),
                (self.num_relations,),
                (batch_size,),
            )
            scores = self.instance.score_r(h=h, all_relations=r, t=t)
            if len(self.cls.relation_shape) == 0:
                exp_shape = (batch_size, 1)
            else:
                exp_shape = (batch_size, self.num_relations)
            self._check_scores(scores=scores, exp_shape=exp_shape)

    def test_score_r_slicing(self):
        """Test score_r with slicing."""
        if len(self.cls.relation_shape) == 0:
            raise unittest.SkipTest("No use in slicing relations for models without relation information.")
        #: The equivalence for models with batch norm only holds in evaluation mode
        self.instance.eval()
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.num_relations,),
            (self.batch_size,),
        )
        scores = self.instance.score_r(h=h, all_relations=r, t=t, slice_size=self.num_relations // 2 + 1)
        scores_no_slice = self.instance.score_r(h=h, all_relations=r, t=t, slice_size=None)
        self._check_close_scores(scores=scores, scores_no_slice=scores_no_slice)

    def test_score_t(self):
        """Test score_t."""
        for batch_size in self._score_batch_sizes:
            h, r, t = self._get_hrt(
                (batch_size,),
                (batch_size,),
                (self.num_entities,),
            )
            scores = self.instance.score_t(h=h, r=r, all_entities=t)
            self._check_scores(scores=scores, exp_shape=(batch_size, self.num_entities))

    def test_score_t_slicing(self):
        """Test score_t with slicing."""
        #: The equivalence for models with batch norm only holds in evaluation mode
        self.instance.eval()
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.num_entities,),
        )
        scores = self.instance.score_t(h=h, r=r, all_entities=t, slice_size=self.num_entities // 2 + 1)
        scores_no_slice = self.instance.score_t(h=h, r=r, all_entities=t, slice_size=None)
        self._check_close_scores(scores=scores, scores_no_slice=scores_no_slice)

    def _check_close_scores(self, scores, scores_no_slice):
        self.assertTrue(torch.isfinite(scores).all(), msg=f'Normal scores had nan:\n\t{scores}')
        self.assertTrue(torch.isfinite(scores_no_slice).all(), msg=f'Slice scores had nan\n\t{scores}')
        self.assertTrue(torch.allclose(scores, scores_no_slice), msg=f'Differences: {scores - scores_no_slice}')

    def _get_test_shapes(self) -> Collection[Tuple[
        Tuple[int, int, int, int],
        Tuple[int, int, int, int],
        Tuple[int, int, int, int],
    ]]:
        """Return a set of test shapes for (h, r, t)."""
        return (
            (  # single score
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            (  # score_r with multi-t
                (self.batch_size, 1, 1, 1),
                (1, 1, self.num_relations, 1),
                (self.batch_size, 1, 1, self.num_entities // 2 + 1),
            ),
            (  # score_r with multi-t and broadcasted head
                (1, 1, 1, 1),
                (1, 1, self.num_relations, 1),
                (self.batch_size, 1, 1, self.num_entities),
            ),
            (  # full cwa
                (1, self.num_entities, 1, 1),
                (1, 1, self.num_relations, 1),
                (1, 1, 1, self.num_entities),
            ),
        )

    def _get_output_shape(
        self,
        hs: Tuple[int, int, int, int],
        rs: Tuple[int, int, int, int],
        ts: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        result = [max(ds) for ds in zip(hs, rs, ts)]
        if len(self.instance.entity_shape) == 0:
            result[1] = result[3] = 1
        if len(self.instance.relation_shape) == 0:
            result[2] = 1
        return tuple(result)

    def test_forward(self):
        """Test forward."""
        for hs, rs, ts in self._get_test_shapes():
            try:
                h, r, t = self._get_hrt(hs, rs, ts)
                scores = self.instance(h=h, r=r, t=t)
                expected_shape = self._get_output_shape(hs, rs, ts)
                self._check_scores(scores=scores, exp_shape=expected_shape)
            except ValueError as error:
                # check whether the error originates from batch norm for single element batches
                small_batch_size = any(s[0] == 1 for s in (hs, rs, ts))
                has_batch_norm = any(
                    isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                    for m in self.instance.modules()
                )
                if small_batch_size and has_batch_norm:
                    logger.warning(
                        f"Skipping test for shapes {hs}, {rs}, {ts} because too small batch size for batch norm",
                    )
                    continue
                raise error

    def test_forward_consistency_with_functional(self):
        """Test forward's consistency with functional."""
        if not isinstance(self.instance, FunctionalInteraction):
            self.skipTest('Not a functional interaction')

        # set in eval mode (otherwise there are non-deterministic factors like Dropout
        self.instance.eval()
        for hs, rs, ts in self._get_test_shapes():
            h, r, t = self._get_hrt(hs, rs, ts)
            scores = self.instance(h=h, r=r, t=t)
            kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)
            scores_f = self.cls.func(**kwargs)
            assert torch.allclose(scores, scores_f)

    def test_scores(self):
        """Test individual scores."""
        # set in eval mode (otherwise there are non-deterministic factors like Dropout
        self.instance.eval()
        for _ in range(10):
            # test multiple different initializations
            self.instance.reset_parameters()
            h, r, t = self._get_hrt((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))

            if isinstance(self.instance, FunctionalInteraction):
                kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)
                # calculate by functional
                scores_f = self.cls.func(**kwargs).view(-1)
            else:
                kwargs = dict(h=h, r=r, t=t)
                scores_f = self.instance(h=h, r=r, t=t)

            # calculate manually
            scores_f_manual = self._exp_score(**kwargs).view(-1)
            assert torch.allclose(scores_f_manual, scores_f), f'Diff: {scores_f_manual - scores_f}'

    @abstractmethod
    def _exp_score(self, **kwargs) -> torch.FloatTensor:
        """Compute the expected score for a single-score batch."""
        raise NotImplementedError(f"{self.cls.__name__}({sorted(kwargs.keys())})")


class TranslationalInteractionTests(InteractionTestCase, ABC):
    """Common tests for translational interaction."""

    kwargs = dict(
        p=2,
    )

    def _additional_score_checks(self, scores):
        assert (scores <= 0).all()


class ResultTrackerTests(GenericTestCase[ResultTracker], unittest.TestCase):
    """Common tests for result trackers."""

    def test_start_run(self):
        """Test start_run."""
        self.instance.start_run(run_name="my_test.run")

    def test_end_run(self):
        """Test end_run."""
        self.instance.end_run()

    def test_log_metrics(self):
        """Test log_metrics."""
        for metrics, step, prefix in (
            (
                # simple
                {"a": 1.0},
                0,
                None,
            ),
            (
                # nested
                {"a": {"b": 5.0}, "c": -1.0},
                2,
                "test",
            ),
        ):
            self.instance.log_metrics(metrics=metrics, step=step, prefix=prefix)

    def test_log_params(self):
        """Test log_params."""
        # nested
        params = {
            "num_epochs": 12,
            "loss": {
                "margin": 2.0,  # a number
                "normalize": True,  # a bool
                "activation": "relu",  # a string
            },
        }
        prefix = None
        self.instance.log_params(params=params, prefix=prefix)


class FileResultTrackerTests(ResultTrackerTests):
    """Tests for FileResultTracker."""

    def setUp(self) -> None:
        """Set up the file result tracker test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temporary_directory.name).joinpath("test.log")
        super().setUp()

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        # prepare a temporary test directory
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["path"] = self.path
        return kwargs

    def tearDown(self) -> None:  # noqa: D102
        # check that file was created
        assert self.path.is_file()
        # make sure to close file before trying to delete it
        self.instance.end_run()
        # delete intermediate files
        self.path.unlink()
        self.temporary_directory.cleanup()


class RegularizerTestCase(GenericTestCase[Regularizer]):
    """A test case for quickly defining common tests for regularizers."""

    #: The batch size
    batch_size: int
    #: The triples factory
    triples_factory: TriplesFactory
    #: Class of regularizer to test
    cls: ClassVar[Type[Regularizer]]
    #: The constructor parameters to pass to the regularizer
    kwargs: ClassVar[Optional[Dict[str, Any]]] = None
    #: The regularizer instance, initialized in setUp
    instance: Regularizer
    #: A positive batch
    positive_batch: MappedTriples
    #: The device
    device: torch.device

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.device = resolve_device()
        self.triples_factory = Nations().training
        self.batch_size = 16
        self.positive_batch = self.triples_factory.mapped_triples[:self.batch_size, :].to(device=self.device)
        super().setUp()
        # move test instance to device
        self.instance = self.instance.to(self.device)

    def test_model(self) -> None:
        """Test whether the regularizer can be passed to a model."""
        # Use RESCAL as it regularizes multiple tensors of different shape.
        model = RESCAL(
            triples_factory=self.triples_factory,
            regularizer=self.instance,
        ).to(self.device)

        # Check if regularizer is stored correctly.
        self.assertEqual(model.regularizer, self.instance)

        # Forward pass (should update regularizer)
        model.score_hrt(hrt_batch=self.positive_batch)

        # Call post_parameter_update (should reset regularizer)
        model.post_parameter_update()

        # Check if regularization term is reset
        self.assertEqual(0., model.regularizer.term)

    def test_reset(self) -> None:
        """Test method `reset`."""
        # Call method
        self.instance.reset()

        self.assertEqual(0., self.instance.regularization_term)

    def test_update(self) -> None:
        """Test method `update`."""
        # Generate random tensors
        a = rand(self.batch_size, 10, generator=self.generator, device=self.device)
        b = rand(self.batch_size, 20, generator=self.generator, device=self.device)

        # Call update
        self.instance.update(a, b)

        # check shape
        self.assertEqual((1,), self.instance.term.shape)

        # compute expected term
        exp_penalties = torch.stack([self._expected_penalty(x) for x in (a, b)])
        expected_term = torch.sum(exp_penalties).view(1) * self.instance.weight
        assert expected_term.shape == (1,)

        self.assertAlmostEqual(self.instance.term.item(), expected_term.item())

    def test_forward(self) -> None:
        """Test the regularizer's `forward` method."""
        # Generate random tensor
        x = rand(self.batch_size, 10, generator=self.generator, device=self.device)

        # calculate penalty
        penalty = self.instance.forward(x=x)

        # check shape
        assert penalty.numel() == 1

        # check value
        expected_penalty = self._expected_penalty(x=x)
        if expected_penalty is None:
            logging.warning(f'{self.__class__.__name__} did not override `_expected_penalty`.')
        else:
            assert (expected_penalty == penalty).all()

    def _expected_penalty(self, x: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        """Compute expected penalty for given tensor."""
        return None


class LpRegularizerTest(RegularizerTestCase):
    """Common test for L_p regularizers."""

    cls = LpRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        p = kwargs.get('p', self.instance.p)
        value = x.norm(p=p, dim=-1).mean()
        if kwargs.get('normalize', False):
            dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
            # FIXME isn't any finite number allowed now?
            if p == 2:
                value = value / dim.sqrt()
            elif p == 1:
                value = value / dim
            else:
                raise NotImplementedError
        return value


class ModelTestCase(unittest_templates.GenericTestCase[Model]):
    """A test case for quickly defining common tests for KGE models."""

    #: Additional arguments passed to the training loop's constructor method
    training_loop_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    #: The triples factory instance
    factory: TriplesFactory

    #: The batch size for use for forward_* tests
    batch_size: int = 20

    #: The embedding dimensionality
    embedding_dim: int = 3

    #: Whether to create inverse triples (needed e.g. by ConvE)
    create_inverse_triples: bool = False

    #: The sampler to use for sLCWA (different e.g. for R-GCN)
    sampler = 'default'

    #: The batch size for use when testing training procedures
    train_batch_size = 400

    #: The number of epochs to train the model
    train_num_epochs = 2

    #: A random number generator from torch
    generator: torch.Generator

    #: The number of parameters which receive a constant (i.e. non-randomized)
    # initialization
    num_constant_init: int = 0

    #: Static extras to append to the CLI
    cli_extras: Sequence[str] = tuple()

    def pre_setup_hook(self) -> None:  # noqa: D102
        # for reproducible testing
        _, self.generator, _ = set_random_seed(42)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        dataset = Nations(create_inverse_triples=self.create_inverse_triples)
        self.factory = dataset.training
        # insert shared parameters
        kwargs["triples_factory"] = self.factory
        kwargs["embedding_dim"] = self.embedding_dim
        return kwargs

    def post_instantiation_hook(self) -> None:  # noqa: D102
        # move model to correct device
        self.instance = self.instance.to_device_()

    def test_get_grad_parameters(self):
        """Test the model's ``get_grad_params()`` method."""
        # assert there is at least one trainable parameter
        assert len(list(self.instance.get_grad_params())) > 0

        # Check that all the parameters actually require a gradient
        for parameter in self.instance.get_grad_params():
            assert parameter.requires_grad

        # Try to initialize an optimizer
        optimizer = SGD(params=self.instance.get_grad_params(), lr=1.0)
        assert optimizer is not None

    def test_reset_parameters_(self):
        """Test :func:`Model.reset_parameters_`."""
        # get model parameters
        params = list(self.instance.parameters())
        old_content = {
            id(p): p.data.detach().clone()
            for p in params
        }

        # re-initialize
        self.instance.reset_parameters_()

        # check that the operation works in-place
        new_params = list(self.instance.parameters())
        assert set(id(np) for np in new_params) == set(id(p) for p in params)

        # check that the parameters where modified
        num_equal_weights_after_re_init = sum(
            1
            for np in new_params
            if (np.data == old_content[id(np)]).all()
        )
        assert num_equal_weights_after_re_init == self.num_constant_init, (
            num_equal_weights_after_re_init, self.num_constant_init,
        )

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        self.assertTrue(torch.all(torch.isfinite(scores)).item(), f'Some scores were not finite:\n{scores}')

        # check whether a gradient can be back-propgated
        scores.mean().backward()

    def test_save(self) -> None:
        """Test that the model can be saved properly."""
        with tempfile.TemporaryDirectory() as temp_directory:
            torch.save(self.instance, os.path.join(temp_directory, 'model.pickle'))

    def test_score_hrt(self) -> None:
        """Test the model's ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.instance.device)
        try:
            scores = self.instance.score_hrt(batch)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(batch, scores)

    def test_score_t(self) -> None:
        """Test the model's ``score_t()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :2].to(self.instance.device)
        # assert batch comprises (head, relation) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores = self.instance.score_t(batch)
        except NotImplementedError:
            self.fail(msg='Score_o not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.instance.num_entities)
        self._check_scores(batch, scores)

    def test_score_h(self) -> None:
        """Test the model's ``score_h()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.instance.device)
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores = self.instance.score_h(batch)
        except NotImplementedError:
            self.fail(msg='Score_s not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.instance.num_entities)
        self._check_scores(batch, scores)

    @pytest.mark.slow
    def test_train_slcwa(self) -> None:
        """Test that sLCWA training does not fail."""
        loop = SLCWATrainingLoop(
            model=self.instance,
            triples_factory=self.factory,
            optimizer=Adagrad(params=self.instance.get_grad_params(), lr=0.001),
            **(self.training_loop_kwargs or {}),
        )
        losses = self._safe_train_loop(
            loop,
            num_epochs=self.train_num_epochs,
            batch_size=self.train_batch_size,
            sampler=self.sampler,
        )
        self.assertIsInstance(losses, list)

    @pytest.mark.slow
    def test_train_lcwa(self) -> None:
        """Test that LCWA training does not fail."""
        loop = LCWATrainingLoop(
            model=self.instance,
            triples_factory=self.factory,
            optimizer=Adagrad(params=self.instance.get_grad_params(), lr=0.001),
            **(self.training_loop_kwargs or {}),
        )
        losses = self._safe_train_loop(
            loop,
            num_epochs=self.train_num_epochs,
            batch_size=self.train_batch_size,
            sampler='default',
        )
        self.assertIsInstance(losses, list)

    def _safe_train_loop(self, loop: TrainingLoop, num_epochs, batch_size, sampler):
        try:
            losses = loop.train(
                triples_factory=self.factory,
                num_epochs=num_epochs,
                batch_size=batch_size,
                sampler=sampler,
                use_tqdm=False,
            )
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        else:
            return losses

    def test_save_load_model_state(self):
        """Test whether a saved model state can be re-loaded."""
        original_model = self.cls(
            random_seed=42,
            **self.instance_kwargs,
        ).to_device_()

        loaded_model = self.cls(
            random_seed=21,
            **self.instance_kwargs,
        ).to_device_()

        def _equal_embeddings(a: RepresentationModule, b: RepresentationModule) -> bool:
            """Test whether two embeddings are equal."""
            return (a(indices=None) == b(indices=None)).all()

        if isinstance(original_model, EntityEmbeddingModel):
            assert not _equal_embeddings(original_model.entity_embeddings, loaded_model.entity_embeddings)
        if isinstance(original_model, EntityRelationEmbeddingModel):
            assert not _equal_embeddings(original_model.relation_embeddings, loaded_model.relation_embeddings)

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, 'test.pt')
            original_model.save_state(path=file_path)
            loaded_model.load_state(path=file_path)
        if isinstance(original_model, EntityEmbeddingModel):
            assert _equal_embeddings(original_model.entity_embeddings, loaded_model.entity_embeddings)
        if isinstance(original_model, EntityRelationEmbeddingModel):
            assert _equal_embeddings(original_model.relation_embeddings, loaded_model.relation_embeddings)

    @property
    def _cli_extras(self):
        """Return a list of extra flags for the CLI."""
        kwargs = self.kwargs or {}
        extras = [
            '--silent',
        ]
        for k, v in kwargs.items():
            extras.append('--' + k.replace('_', '-'))
            extras.append(str(v))

        # For the high/low memory test cases of NTN, SE, etc.
        if self.training_loop_kwargs and 'automatic_memory_optimization' in self.training_loop_kwargs:
            automatic_memory_optimization = self.training_loop_kwargs.get('automatic_memory_optimization')
            if automatic_memory_optimization is True:
                extras.append('--automatic-memory-optimization')
            elif automatic_memory_optimization is False:
                extras.append('--no-automatic-memory-optimization')
            # else, leave to default

        extras += [
            '--number-epochs', self.train_num_epochs,
            '--embedding-dim', self.embedding_dim,
            '--batch-size', self.train_batch_size,
        ]
        extras.extend(self.cli_extras)
        # TODO: Make sure that inverse triples are created if create_inverse_triples=True
        extras = [str(e) for e in extras]
        return extras

    @pytest.mark.slow
    def test_cli_training_nations(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH] + self._cli_extras)

    @pytest.mark.slow
    def test_cli_training_kinships(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', KINSHIPS_TRAIN_PATH] + self._cli_extras)

    @pytest.mark.slow
    def test_cli_training_nations_testing(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH, '-q', NATIONS_TEST_PATH] + self._cli_extras)

    def _help_test_cli(self, args):
        """Test running the pipeline on all models."""
        if issubclass(self.cls, pykeen.models.RGCN) or self.cls is pykeen.models.ERModel:
            self.skipTest(f"Cannot choose interaction via CLI for {self.cls}.")
        runner = CliRunner()
        cli = build_cli_from_cls(self.cls)
        # TODO: Catch HolE MKL error?
        result: Result = runner.invoke(cli, args)

        self.assertEqual(
            0,
            result.exit_code,
            msg=f'''
Command
=======
$ pykeen train {self.cls.__name__.lower()} {' '.join(map(str, args))}

Output
======
{result.output}

Exception
=========
{result.exc_info[1]}

Traceback
=========
{''.join(traceback.format_tb(result.exc_info[2]))}
            ''',
        )

    def test_has_hpo_defaults(self):
        """Test that there are defaults for HPO."""
        try:
            d = self.cls.hpo_default
        except AttributeError:
            self.fail(msg=f'{self.cls.__name__} is missing hpo_default class attribute')
        else:
            self.assertIsInstance(d, dict)

    def test_post_parameter_update_regularizer(self):
        """Test whether post_parameter_update resets the regularization term."""
        if not hasattr(self.instance, 'regularizer'):
            self.skipTest('no regularizer')

        # set regularizer term to something that isn't zero
        self.instance.regularizer.regularization_term = torch.ones(1, dtype=torch.float, device=self.instance.device)

        # call post_parameter_update
        self.instance.post_parameter_update()

        # assert that the regularization term has been reset
        expected_term = torch.zeros(1, dtype=torch.float, device=self.instance.device)
        assert self.instance.regularizer.regularization_term == expected_term

    def test_post_parameter_update(self):
        """Test whether post_parameter_update correctly enforces model constraints."""
        # do one optimization step
        opt = optim.SGD(params=self.instance.parameters(), lr=1.)
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.instance.device)
        scores = self.instance.score_hrt(hrt_batch=batch)
        fake_loss = scores.mean()
        fake_loss.backward()
        opt.step()

        # call post_parameter_update
        self.instance.post_parameter_update()

        # check model constraints
        self._check_constraints()

    def _check_constraints(self):
        """Check model constraints."""

    def test_score_h_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_h()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.instance.device)
        self.instance.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_h = self.instance.score_h(batch)
            scores_hrt = super(self.instance.__class__, self.instance).score_h(batch)
        except NotImplementedError:
            self.fail(msg='Score_h not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        assert torch.allclose(scores_h, scores_hrt, atol=1e-06)

    def test_score_r_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_r()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, [0, 2]].to(self.instance.device)
        self.instance.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_r = self.instance.score_r(batch)
            scores_hrt = super(self.instance.__class__, self.instance).score_r(batch)
        except NotImplementedError:
            self.fail(msg='Score_h not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        assert torch.allclose(scores_r, scores_hrt, atol=1e-06)

    def test_score_t_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_t()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :-1].to(self.instance.device)
        self.instance.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores_t = self.instance.score_t(batch)
            scores_hrt = super(self.instance.__class__, self.instance).score_t(batch)
        except NotImplementedError:
            self.fail(msg='Score_h not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e

        assert torch.allclose(scores_t, scores_hrt, atol=1e-06)

    def test_reset_parameters_constructor_call(self):
        """Tests whether reset_parameters is called in the constructor."""
        with patch.object(self.cls, 'reset_parameters_', return_value=None) as mock_method:
            try:
                self.cls(**self.instance_kwargs)
            except TypeError as error:
                assert error.args == ("'NoneType' object is not callable",)
            mock_method.assert_called_once()

    def test_custom_representations(self):
        """Tests whether we can provide custom representations."""
        if isinstance(self.instance, EntityEmbeddingModel):
            old_embeddings = self.instance.entity_embeddings
            self.instance.entity_embeddings = CustomRepresentations(
                num_entities=self.factory.num_entities,
                shape=old_embeddings.shape,
            )
            # call some functions
            self.instance.reset_parameters_()
            self.test_score_hrt()
            self.test_score_t()
            # reset to old state
            self.instance.entity_embeddings = old_embeddings
        elif isinstance(self.instance, EntityRelationEmbeddingModel):
            old_embeddings = self.instance.relation_embeddings
            self.instance.relation_embeddings = CustomRepresentations(
                num_entities=self.factory.num_relations,
                shape=old_embeddings.shape,
            )
            # call some functions
            self.instance.reset_parameters_()
            self.test_score_hrt()
            self.test_score_t()
            # reset to old state
            self.instance.relation_embeddings = old_embeddings
        else:
            self.skipTest(f'Not testing custom representations for model: {self.instance.__class__.__name__}')


class DistanceModelTestCase(ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class BaseKG2ETest(ModelTestCase):
    """General tests for the KG2E model."""

    cls = pykeen.models.KG2E

    def _check_constraints(self):
        """Check model constraints.

        * Entity and relation embeddings have to have at most unit L2 norm.
        * Covariances have to have values between c_min and c_max
        """
        for embedding in (self.instance.entity_embeddings, self.instance.relation_embeddings):
            assert all_in_bounds(embedding(indices=None).norm(p=2, dim=-1), high=1., a_tol=EPSILON)
        for cov in (self.instance.entity_covariances, self.instance.relation_covariances):
            assert all_in_bounds(cov(indices=None), low=self.instance.c_min, high=self.instance.c_max)


class BaseNTNTest(ModelTestCase):
    """Test the NTN model."""

    cls = pykeen.models.NTN

    def test_can_slice(self):
        """Test that the slicing properties are calculated correctly."""
        self.assertTrue(self.instance.can_slice_h)
        self.assertFalse(self.instance.can_slice_r)
        self.assertTrue(self.instance.can_slice_t)


class BaseRGCNTest(ModelTestCase):
    """Test the R-GCN model."""

    cls = pykeen.models.RGCN
    sampler = 'schlichtkrull'

    def _check_constraints(self):
        """Check model constraints.

        Enriched embeddings have to be reset.
        """
        assert self.instance.entity_representations[0].enriched_embeddings is None


class RepresentationTestCase(GenericTestCase[RepresentationModule]):
    """Common tests for representation modules."""

    batch_size: int = 2
    num_negatives: int = 3

    def _check_result(self, x: torch.FloatTensor, prefix_shape: Tuple[int, ...]):
        """Check the result."""
        # check type
        assert torch.is_tensor(x)
        assert x.dtype == torch.get_default_dtype()

        # check shape
        expected_shape = prefix_shape + self.instance.shape
        self.assertEqual(x.shape, expected_shape)

    def _test_forward(self, indices: Optional[torch.LongTensor]):
        """Test forward method."""
        representations = self.instance.forward(indices=indices)
        prefix_shape = (self.instance.max_id,) if indices is None else tuple(indices.shape)
        self._check_result(x=representations, prefix_shape=prefix_shape)

    def _test_canonical_shape(self, indices: Optional[torch.LongTensor]):
        """Test canonical shape."""
        x = self.instance.get_in_canonical_shape(indices=indices)
        if indices is None:
            prefix_shape = (1, self.instance.max_id)
        elif indices.ndimension() == 1:
            prefix_shape = (indices.shape[0], 1)
        elif indices.ndimension() == 2:
            prefix_shape = tuple(indices.shape)
        else:
            raise AssertionError(indices.shape)
        self._check_result(x=x, prefix_shape=prefix_shape)

    def _test_more_canonical_shape(self, indices: Optional[torch.LongTensor]):
        """Test more canonical shape."""
        for i, dim in enumerate(("h", "r", "t"), start=1):
            x = self.instance.get_in_more_canonical_shape(dim=dim, indices=indices)
            prefix_shape = [1, 1, 1, 1]
            if indices is None:
                prefix_shape[i] = self.instance.max_id
            elif indices.ndimension() == 1:
                prefix_shape[0] = indices.shape[0]
            elif indices.ndimension() == 2:
                prefix_shape[0] = indices.shape[0]
                prefix_shape[i] = indices.shape[1]
            else:
                raise AssertionError(indices.shape)
            self._check_result(x=x, prefix_shape=tuple(prefix_shape))

    def _test_indices(self, indices: Optional[torch.LongTensor]):
        """Test forward and canonical shape for indices."""
        self._test_forward(indices=indices)
        self._test_canonical_shape(indices=indices)
        self._test_more_canonical_shape(indices=indices)

    def test_no_indices(self):
        """Test without indices."""
        self._test_indices(indices=None)

    def test_1d_indices(self):
        """Test with 1-dimensional indices."""
        self._test_indices(indices=torch.randint(self.instance.max_id, size=(self.batch_size,)))

    def test_2d_indices(self):
        """Test with 1-dimensional indices."""
        self._test_indices(indices=(torch.randint(self.instance.max_id, size=(self.batch_size, self.num_negatives))))

    def test_all_indices(self):
        """Test with all indices."""
        self._test_indices(indices=torch.arange(self.instance.max_id))


class EdgeWeightingTestCase(GenericTestCase[pykeen.nn.weighting.EdgeWeighting]):
    """Tests for message weighting."""

    #: The number of entities
    num_entities: int = 16

    #: The number of triples
    num_triples: int = 101

    def post_instantiation_hook(self):  # noqa: D102
        self.source, self.target = torch.randint(self.num_entities, size=(2, self.num_triples))

    def test_message_weighting(self):
        """Perform common tests for message weighting."""
        weights = self.instance(source=self.source, target=self.target)

        # check shape
        assert weights.shape == self.source.shape

        # check dtype
        assert weights.dtype == torch.float32

        # check finite values (e.g. due to division by zero)
        assert torch.isfinite(weights).all()

        # check non-negativity
        assert (weights >= 0.).all()


class DecompositionTestCase(GenericTestCase[pykeen.nn.message_passing.Decomposition]):
    """Tests for relation-specific weight decomposition message passing classes."""

    #: The input dimension
    input_dim: int = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        self.output_dim = self.input_dim
        self.factory = Nations().training
        self.source, self.edge_type, self.target = self.factory.mapped_triples.t()
        self.x = torch.rand(self.factory.num_entities, self.input_dim)
        kwargs["input_dim"] = self.input_dim
        kwargs["num_relations"] = self.factory.num_relations
        return kwargs

    def test_forward(self):
        """Test the :meth:`Decomposition.forward` function."""
        for node_keep_mask in [None, torch.rand(size=(self.factory.num_entities,)) < 0.5]:
            for edge_weights in [None, torch.rand_like(self.source, dtype=torch.get_default_dtype())]:
                y = self.instance(
                    x=self.x,
                    node_keep_mask=node_keep_mask,
                    source=self.source,
                    target=self.target,
                    edge_type=self.edge_type,
                    edge_weights=edge_weights,
                )
                assert y.shape == (self.x.shape[0], self.output_dim)


class BasesDecompositionTestCase(DecompositionTestCase):
    """Tests for bases Decomposition."""

    cls = pykeen.nn.message_passing.BasesDecomposition


class LiteralTestCase(InteractionTestCase):
    """Tests for literal ineractions."""

    cls = LiteralInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h_proj = self.instance.combination(*h)
        t_proj = self.instance.combination(*t)
        return self.instance.base(h_proj, r, t_proj)
