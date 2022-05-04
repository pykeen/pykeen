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
from collections import Counter
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest.case import SkipTest
from unittest.mock import patch

import numpy
import numpy.random
import pandas
import pytest
import torch
import torch.utils.data
import unittest_templates
from class_resolver import HintOrType
from click.testing import CliRunner, Result
from docdata import get_docdata
from torch import optim
from torch.nn import functional
from torch.optim import SGD, Adagrad

import pykeen.models
import pykeen.nn.message_passing
import pykeen.nn.node_piece
import pykeen.nn.representation
import pykeen.nn.weighting
from pykeen.datasets import Nations
from pykeen.datasets.base import LazyDataset
from pykeen.datasets.ea.combination import GraphPairCombinator
from pykeen.datasets.kinships import KINSHIPS_TRAIN_PATH
from pykeen.datasets.mocks import create_inductive_dataset
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH
from pykeen.evaluation import Evaluator, MetricResults, evaluator_resolver
from pykeen.losses import Loss, PairwiseLoss, PointwiseLoss, SetwiseLoss, UnsupportedLabelSmoothingError
from pykeen.metrics import rank_based_metric_resolver
from pykeen.metrics.ranking import (
    DerivedRankBasedMetric,
    NoClosedFormError,
    RankBasedMetric,
    generate_num_candidates_and_ranks,
)
from pykeen.models import RESCAL, EntityRelationEmbeddingModel, Model, TransE
from pykeen.models.cli import build_cli_from_cls
from pykeen.models.nbase import ERModel
from pykeen.nn.modules import DistMultInteraction, FunctionalInteraction, Interaction, LiteralInteraction
from pykeen.nn.representation import Representation
from pykeen.optimizers import optimizer_resolver
from pykeen.pipeline import pipeline
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.trackers import ResultTracker
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop, TrainingCallback, TrainingLoop
from pykeen.triples import Instances, TriplesFactory, generation
from pykeen.triples.instances import BaseBatchedSLCWAInstances, SLCWABatch
from pykeen.triples.splitting import Cleaner, Splitter
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.triples.utils import get_entities
from pykeen.typing import (
    EA_SIDE_LEFT,
    EA_SIDE_RIGHT,
    LABEL_HEAD,
    LABEL_TAIL,
    TRAINING,
    HeadRepresentation,
    InductiveMode,
    Initializer,
    MappedTriples,
    RelationRepresentation,
    TailRepresentation,
)
from pykeen.utils import (
    all_in_bounds,
    get_batchnorm_modules,
    getattr_or_docdata,
    is_triple_tensor_subset,
    resolve_device,
    set_random_seed,
    triple_tensor_to_set,
    unpack_singletons,
)
from tests.constants import EPSILON
from tests.mocks import CustomRepresentation
from tests.utils import rand

try:
    import torch_geometric
except ImportError:
    torch_geometric = None

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
            triples_factory.num_triples
            for triples_factory in (self.dataset._training, self.dataset._testing, self.dataset._validation)
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
        self.assertAlmostEqual(start, end, delta=1.0e-02, msg="Caching should have made this operation fast")

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
        positive_scores = torch.zeros(self.batch_size, 1, requires_grad=True)
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


class GMRLTestCase(PairwiseLossTestCase):
    """Tests for generalized margin ranking loss."""

    def test_label_smoothing_raise(self):
        """Test errors are raised if label smoothing is given."""
        with self.assertRaises(UnsupportedLabelSmoothingError):
            self.instance.process_lcwa_scores(..., ..., label_smoothing=5)
        with self.assertRaises(UnsupportedLabelSmoothingError):
            self.instance.process_lcwa_scores(..., ..., label_smoothing=5)


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
    dtype: torch.dtype = torch.get_default_dtype()

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
                torch.rand(
                    size=tuple(prefix_shape) + tuple(shape_kwargs[dim] for dim in weight_shape),
                    requires_grad=True,
                    dtype=self.dtype,
                )
                for weight_shape in weight_shapes
            )
            for prefix_shape, weight_shapes in zip(
                shapes,
                [self.instance.entity_shape, self.instance.relation_shape, self.instance.tail_entity_shape],
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
        self.assertTrue(torch.isfinite(scores).all(), msg=f"Normal scores had nan:\n\t{scores}")
        self.assertTrue(torch.isfinite(scores_no_slice).all(), msg=f"Slice scores had nan\n\t{scores}")
        self.assertTrue(torch.allclose(scores, scores_no_slice), msg=f"Differences: {scores - scores_no_slice}")

    def _get_test_shapes(self) -> Collection[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
        """Return a set of test shapes for (h, r, t)."""
        return (
            (  # single score
                tuple(),
                tuple(),
                tuple(),
            ),
            (  # score_r with multi-t
                (self.batch_size, 1, 1),
                (1, self.num_relations, 1),
                (self.batch_size, 1, self.num_entities // 2 + 1),
            ),
            (  # score_r with multi-t and broadcasted head
                (1, 1, 1),
                (1, self.num_relations, 1),
                (self.batch_size, 1, self.num_entities),
            ),
            (  # full cwa
                (self.num_entities, 1, 1),
                (1, self.num_relations, 1),
                (1, 1, self.num_entities),
            ),
        )

    def _get_output_shape(
        self,
        hs: Tuple[int, ...],
        rs: Tuple[int, ...],
        ts: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        components = []
        if self.instance.entity_shape:
            components.extend((hs, ts))
        if self.instance.relation_shape:
            components.append(rs)
        return tuple(max(ds) for ds in zip(*components))

    def test_forward(self):
        """Test forward."""
        for hs, rs, ts in self._get_test_shapes():
            if get_batchnorm_modules(self.instance) and any(numpy.prod(s) == 1 for s in (hs, rs, ts)):
                logger.warning(
                    f"Skipping test for shapes {hs}, {rs}, {ts} because too small batch size for batch norm",
                )
                continue
            h, r, t = self._get_hrt(hs, rs, ts)
            scores = self.instance(h=h, r=r, t=t)
            expected_shape = self._get_output_shape(hs, rs, ts)
            self._check_scores(scores=scores, exp_shape=expected_shape)

    def test_forward_consistency_with_functional(self):
        """Test forward's consistency with functional."""
        if not isinstance(self.instance, FunctionalInteraction):
            self.skipTest("Not a functional interaction")

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
            h, r, t = self._get_hrt(tuple(), tuple(), tuple())

            if isinstance(self.instance, FunctionalInteraction):
                kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)
                # calculate by functional
                scores_f = self.cls.func(**kwargs).view(-1)
            else:
                kwargs = dict(h=h, r=r, t=t)
                scores_f = self.instance(h=h, r=r, t=t)

            # calculate manually
            scores_f_manual = self._exp_score(**kwargs).view(-1)
            assert torch.allclose(scores_f_manual, scores_f), f"Diff: {scores_f_manual - scores_f}"

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
        self.positive_batch = self.triples_factory.mapped_triples[: self.batch_size, :].to(device=self.device)
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
        self.assertEqual(0.0, model.regularizer.term)

    def test_reset(self) -> None:
        """Test method `reset`."""
        # Call method
        self.instance.reset()

        self.assertEqual(0.0, self.instance.regularization_term)

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
            logging.warning(f"{self.__class__.__name__} did not override `_expected_penalty`.")
        else:
            assert (expected_penalty == penalty).all()

    def _expected_penalty(self, x: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        """Compute expected penalty for given tensor."""
        return None

    def test_pop_regularization_term(self):
        """Verify popping a regularization term."""
        # update term
        x = torch.rand(self.batch_size, 10, generator=self.generator, device=self.device, requires_grad=True)
        self.instance.update(x)

        # check that the expected term is returned
        exp = (self.instance.weight * self.instance.regularization_term).item()
        self.assertEqual(exp, self.instance.pop_regularization_term().item())


class LpRegularizerTest(RegularizerTestCase):
    """Common test for L_p regularizers."""

    cls = LpRegularizer

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        p = kwargs.get("p", self.instance.p)
        value = x.norm(p=p, dim=-1).mean()
        if kwargs.get("normalize", False):
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
    sampler: Optional[str] = None

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

    #: the model's device
    device: torch.device

    #: the inductive mode
    mode: ClassVar[Optional[InductiveMode]] = None

    def pre_setup_hook(self) -> None:  # noqa: D102
        # for reproducible testing
        _, self.generator, _ = set_random_seed(42)
        self.device = resolve_device()

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
        self.instance = self.instance.to(self.device)

    def test_get_grad_parameters(self):
        """Test the model's ``get_grad_params()`` method."""
        self.assertLess(
            0, len(list(self.instance.get_grad_params())), msg="There is not at least one trainable parameter"
        )

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
        old_content = {id(p): p.data.detach().clone() for p in params}

        # re-initialize
        self.instance.reset_parameters_()

        # check that the operation works in-place
        new_params = list(self.instance.parameters())
        assert set(id(np) for np in new_params) == set(id(p) for p in params)

        # check that the parameters where modified
        num_equal_weights_after_re_init = sum(
            1 for new_param in new_params if (new_param.data == old_content[id(new_param)]).all()
        )
        self.assertEqual(num_equal_weights_after_re_init, self.num_constant_init)

    def _check_scores(self, batch, scores) -> None:
        """Check the scores produced by a forward function."""
        # check for finite values by default
        self.assertTrue(torch.all(torch.isfinite(scores)).item(), f"Some scores were not finite:\n{scores}")

        # check whether a gradient can be back-propgated
        scores.mean().backward()

    def test_save(self) -> None:
        """Test that the model can be saved properly."""
        with tempfile.TemporaryDirectory() as temp_directory:
            torch.save(self.instance, os.path.join(temp_directory, "model.pickle"))

    def test_score_hrt(self) -> None:
        """Test the model's ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[: self.batch_size, :].to(self.instance.device)
        try:
            scores = self.instance.score_hrt(batch, mode=self.mode)
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
                self.skipTest(str(e))
            else:
                raise e
        self.assertEqual(scores.shape, (self.batch_size, 1))
        self._check_scores(batch, scores)

    def test_score_t(self) -> None:
        """Test the model's ``score_t()`` function."""
        batch = self.factory.mapped_triples[: self.batch_size, :2].to(self.instance.device)
        # assert batch comprises (head, relation) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores = self.instance.score_t(batch, mode=self.mode)
        except NotImplementedError:
            self.fail(msg="score_t not yet implemented")
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.instance.num_entities)
        self._check_scores(batch, scores)

    def test_score_r(self) -> None:
        """Test the model's ``score_r()`` function."""
        batch = self.factory.mapped_triples[: self.batch_size, [0, 2]].to(self.instance.device)
        # assert batch comprises (head, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch < self.factory.num_entities).all()
        try:
            scores = self.instance.score_r(batch, mode=self.mode)
        except NotImplementedError:
            self.fail(msg="score_r not yet implemented")
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
                self.skipTest(str(e))
            else:
                raise e
        if self.create_inverse_triples:
            # TODO: look into score_r for inverse relations
            logger.warning("score_r's shape is not clear yet for models with inverse relations")
        else:
            assert scores.shape == (self.batch_size, self.instance.num_relations)
        self._check_scores(batch, scores)

    def test_score_h(self) -> None:
        """Test the model's ``score_h()`` function."""
        batch = self.factory.mapped_triples[: self.batch_size, 1:].to(self.instance.device)
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores = self.instance.score_h(batch, mode=self.mode)
        except NotImplementedError:
            self.fail(msg="score_h not yet implemented")
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
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
            sampler=None,
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
            if str(e) == "fft: ATen not compiled with MKL support":
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
        )

        loaded_model = self.cls(
            random_seed=21,
            **self.instance_kwargs,
        )

        def _equal_embeddings(a: Representation, b: Representation) -> bool:
            """Test whether two embeddings are equal."""
            return (a(indices=None) == b(indices=None)).all()

        if isinstance(original_model, EntityRelationEmbeddingModel):
            assert not _equal_embeddings(original_model.relation_embeddings, loaded_model.relation_embeddings)

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "test.pt")
            original_model.save_state(path=file_path)
            loaded_model.load_state(path=file_path)
        if isinstance(original_model, EntityRelationEmbeddingModel):
            assert _equal_embeddings(original_model.relation_embeddings, loaded_model.relation_embeddings)

    @property
    def _cli_extras(self):
        """Return a list of extra flags for the CLI."""
        kwargs = self.kwargs or {}
        extras = [
            "--silent",
        ]
        for k, v in kwargs.items():
            extras.append("--" + k.replace("_", "-"))
            extras.append(str(v))

        # For the high/low memory test cases of NTN, SE, etc.
        if self.training_loop_kwargs and "automatic_memory_optimization" in self.training_loop_kwargs:
            automatic_memory_optimization = self.training_loop_kwargs.get("automatic_memory_optimization")
            if automatic_memory_optimization is True:
                extras.append("--automatic-memory-optimization")
            elif automatic_memory_optimization is False:
                extras.append("--no-automatic-memory-optimization")
            # else, leave to default

        extras += [
            "--number-epochs",
            self.train_num_epochs,
            "--embedding-dim",
            self.embedding_dim,
            "--batch-size",
            self.train_batch_size,
        ]
        extras.extend(self.cli_extras)

        # Make sure that inverse triples are created if create_inverse_triples=True
        if self.create_inverse_triples:
            extras.append("--create-inverse-triples")

        extras = [str(e) for e in extras]
        return extras

    @pytest.mark.slow
    def test_cli_training_nations(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(["-t", NATIONS_TRAIN_PATH] + self._cli_extras)

    @pytest.mark.slow
    def test_pipeline_nations_early_stopper(self):
        """Test running the pipeline with early stopping."""
        model_kwargs = dict(self.instance_kwargs)
        # triples factory is added by the pipeline
        model_kwargs.pop("triples_factory")
        pipeline(
            model=self.cls,
            model_kwargs=model_kwargs,
            dataset="nations",
            dataset_kwargs=dict(create_inverse_triples=self.create_inverse_triples),
            stopper="early",
            training_loop_kwargs=self.training_loop_kwargs,
            stopper_kwargs=dict(frequency=1),
            training_kwargs=dict(
                batch_size=self.train_batch_size,
                num_epochs=self.train_num_epochs,
            ),
        )

    @pytest.mark.slow
    def test_cli_training_kinships(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(["-t", KINSHIPS_TRAIN_PATH] + self._cli_extras)

    @pytest.mark.slow
    def test_cli_training_nations_testing(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(["-t", NATIONS_TRAIN_PATH, "-q", NATIONS_TEST_PATH] + self._cli_extras)

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
            msg=f"""
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
            """,
        )

    def test_has_hpo_defaults(self):
        """Test that there are defaults for HPO."""
        try:
            d = self.cls.hpo_default
        except AttributeError:
            self.fail(msg=f"{self.cls.__name__} is missing hpo_default class attribute")
        else:
            self.assertIsInstance(d, dict)

    def test_post_parameter_update_regularizer(self):
        """Test whether post_parameter_update resets the regularization term."""
        if not hasattr(self.instance, "regularizer"):
            self.skipTest("no regularizer")

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
        opt = optim.SGD(params=self.instance.parameters(), lr=1.0)
        batch = self.factory.mapped_triples[: self.batch_size, :].to(self.instance.device)
        scores = self.instance.score_hrt(hrt_batch=batch, mode=self.mode)
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
        if isinstance(self.instance, ERModel):
            raise SkipTest("ERModel fulfils this by design.")
        batch = self.factory.mapped_triples[: self.batch_size, 1:].to(self.instance.device)
        self.instance.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_h = self.instance.score_h(batch)
            scores_hrt = super(self.instance.__class__, self.instance).score_h(batch)
        except NotImplementedError:
            self.fail(msg="Score_h not yet implemented")
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
                self.skipTest(str(e))
            else:
                raise e

        self.assertIsNotNone(scores_hrt)
        assert torch.allclose(scores_h, scores_hrt, atol=1e-06)

    def test_score_r_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_r()`` and ``score_hrt()`` function."""
        if isinstance(self.instance, ERModel):
            raise SkipTest("ERModel fulfils this by design.")
        batch = self.factory.mapped_triples[: self.batch_size, [0, 2]].to(self.instance.device)
        self.instance.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_r = self.instance.score_r(batch)
            scores_hrt = super(self.instance.__class__, self.instance).score_r(batch)
        except NotImplementedError:
            self.fail(msg="Score_h not yet implemented")
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
                self.skipTest(str(e))
            else:
                raise e

        self.assertIsNotNone(scores_hrt)
        assert torch.allclose(scores_r, scores_hrt, atol=1e-06)

    def test_score_t_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_t()`` and ``score_hrt()`` function."""
        if isinstance(self.instance, ERModel):
            raise SkipTest("ERModel fulfils this by design.")
        batch = self.factory.mapped_triples[: self.batch_size, :-1].to(self.instance.device)
        self.instance.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores_t = self.instance.score_t(batch)
            scores_hrt = super(self.instance.__class__, self.instance).score_t(batch)
        except NotImplementedError:
            self.fail(msg="Score_h not yet implemented")
        except RuntimeError as e:
            if str(e) == "fft: ATen not compiled with MKL support":
                self.skipTest(str(e))
            else:
                raise e

        self.assertIsNotNone(scores_hrt)
        assert torch.allclose(scores_t, scores_hrt, atol=1e-06)

    def test_reset_parameters_constructor_call(self):
        """Tests whether reset_parameters is called in the constructor."""
        with patch.object(self.cls, "reset_parameters_", return_value=None) as mock_method:
            try:
                self.cls(**self.instance_kwargs)
            except TypeError as error:
                assert error.args == ("'NoneType' object is not callable",)
            mock_method.assert_called_once()

    def test_custom_representations(self):
        """Tests whether we can provide custom representations."""
        if isinstance(self.instance, EntityRelationEmbeddingModel):
            old_embeddings = self.instance.relation_embeddings
            self.instance.relation_embeddings = CustomRepresentation(
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
            self.skipTest(f"Not testing custom representations for model: {self.instance.__class__.__name__}")


class DistanceModelTestCase(ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class BaseKG2ETest(ModelTestCase):
    """General tests for the KG2E model."""

    cls = pykeen.models.KG2E
    c_min: float = 0.01
    c_max: float = 1.0

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["c_min"] = self.c_min
        kwargs["c_max"] = self.c_max
        return kwargs

    def _check_constraints(self):
        """Check model constraints.

        * Entity and relation embeddings have to have at most unit L2 norm.
        * Covariances have to have values between c_min and c_max
        """
        self.instance: ERModel
        (e_mean, e_cov), (r_mean, r_cov) = self.instance.entity_representations, self.instance.relation_representations
        for embedding in (e_mean, r_mean):
            assert all_in_bounds(embedding(indices=None).norm(p=2, dim=-1), high=1.0, a_tol=EPSILON)
        for cov in (e_cov, r_cov):
            assert all_in_bounds(
                cov(indices=None), low=self.instance_kwargs["c_min"], high=self.instance_kwargs["c_max"]
            )


class BaseRGCNTest(ModelTestCase):
    """Test the R-GCN model."""

    cls = pykeen.models.RGCN
    sampler = "schlichtkrull"

    def _check_constraints(self):
        """Check model constraints.

        Enriched embeddings have to be reset.
        """
        assert self.instance.entity_representations[0].enriched_embeddings is None


class BaseNodePieceTest(ModelTestCase):
    """Test the NodePiece model."""

    cls = pykeen.models.NodePiece
    create_inverse_triples = True

    def _help_test_cli(self, args):  # noqa: D102
        if self.instance_kwargs.get("tokenizers_kwargs"):
            raise SkipTest("No support for tokenizers_kwargs via CLI.")
        return super()._help_test_cli(args)


class InductiveModelTestCase(ModelTestCase):
    """Tests for inductive models."""

    mode = TRAINING
    num_relations: ClassVar[int] = 7
    num_entities_transductive: ClassVar[int] = 13
    num_entities_inductive: ClassVar[int] = 5
    num_triples_training: ClassVar[int] = 33
    num_triples_inference: ClassVar[int] = 31
    num_triples_testing: ClassVar[int] = 37

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        dataset = create_inductive_dataset(
            num_relations=self.num_relations,
            num_entities_transductive=self.num_entities_transductive,
            num_entities_inductive=self.num_entities_inductive,
            num_triples_training=self.num_triples_training,
            num_triples_inference=self.num_triples_inference,
            num_triples_testing=self.num_triples_testing,
            create_inverse_triples=self.create_inverse_triples,
        )
        training_loop_kwargs = dict(self.training_loop_kwargs or dict())
        training_loop_kwargs["mode"] = self.mode
        InductiveModelTestCase.training_loop_kwargs = training_loop_kwargs
        # dataset = InductiveFB15k237(create_inverse_triples=self.create_inverse_triples)
        kwargs["triples_factory"] = self.factory = dataset.transductive_training
        kwargs["inference_factory"] = dataset.inductive_inference
        return kwargs

    def _help_test_cli(self, args):  # noqa: D102
        raise SkipTest("Inductive models are not compatible the CLI.")

    def test_pipeline_nations_early_stopper(self):  # noqa: D102
        raise SkipTest("Inductive models are not compatible the pipeline.")


class RepresentationTestCase(GenericTestCase[Representation]):
    """Common tests for representation modules."""

    batch_size: ClassVar[int] = 2
    num_negatives: ClassVar[int] = 3

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
        representations = self.instance.forward_unique(indices=indices)
        prefix_shape = (self.instance.max_id,) if indices is None else tuple(indices.shape)
        self._check_result(x=representations, prefix_shape=prefix_shape)

    def _test_indices(self, indices: Optional[torch.LongTensor]):
        """Test forward and canonical shape for indices."""
        self._test_forward(indices=indices)

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

    def test_dropout(self):
        """Test dropout layer."""
        # create a new instance with guaranteed dropout
        kwargs = self.instance_kwargs
        kwargs.pop("dropout", None)
        dropout_instance = self.cls(**kwargs, dropout=0.1)
        # set to training mode
        dropout_instance.train()
        # check for different output
        indices = torch.arange(2)
        # use more samples to make sure that enough values can be dropped
        a = torch.stack([dropout_instance(indices) for _ in range(20)])
        assert not (a[0:1] == a).all()


class TriplesFactoryRepresentationTestCase(RepresentationTestCase):
    """Tests for representations requiring triples factories."""

    num_entities: ClassVar[int] = 8
    num_relations: ClassVar[int] = 7
    num_triples: ClassVar[int] = 31
    create_inverse_triples: bool = False

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = generation.generate_triples_factory(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_triples=self.num_triples,
            create_inverse_triples=self.create_inverse_triples,
        )
        return kwargs


@unittest.skipIf(torch_geometric is None, "Need to install `torch_geometric`")
class MessagePassingRepresentationTests(TriplesFactoryRepresentationTestCase):
    """Tests for message passing representations."""

    def test_consistency_k_hop(self):
        """Test consistency of results between using only k-hop and using the full graph."""
        # select random indices
        indices = torch.randint(self.num_entities, size=(self.num_entities // 2,), generator=self.generator)
        assert isinstance(self.instance, pykeen.nn.pyg.MessagePassingRepresentation)
        # forward pass with full graph
        self.instance.restrict_k_hop = False
        x_full = self.instance(indices=indices)
        # forward pass with restricted graph
        self.instance.restrict_k_hop = True
        x_restrict = self.instance(indices=indices)
        # verify the results are similar
        assert torch.allclose(x_full, x_restrict)


class EdgeWeightingTestCase(GenericTestCase[pykeen.nn.weighting.EdgeWeighting]):
    """Tests for message weighting."""

    #: The number of entities
    num_entities: int = 16

    #: The number of triples
    num_triples: int = 101

    #: the message dim
    message_dim: int = 3

    def post_instantiation_hook(self):  # noqa: D102
        self.source, self.target = torch.randint(self.num_entities, size=(2, self.num_triples))
        self.message = torch.rand(self.num_triples, self.message_dim, requires_grad=True)
        # TODO: separation message vs. entity dim?
        self.x_e = torch.rand(self.num_entities, self.message_dim)

    def _test(self, weights: torch.FloatTensor, shape: Tuple[int, ...]):
        """Perform common tests."""
        # check shape
        assert weights.shape == shape

        # check dtype
        assert weights.dtype == torch.float32

        # check finite values (e.g. due to division by zero)
        assert torch.isfinite(weights).all()

        # check non-negativity
        assert (weights >= 0.0).all()

    def test_message_weighting(self):
        """Test message weighting with message."""
        self._test(
            weights=self.instance(source=self.source, target=self.target, message=self.message, x_e=self.x_e),
            shape=self.message.shape,
        )

    def test_message_weighting_no_message(self):
        """Test message weighting without message."""
        if self.instance.needs_message:
            raise SkipTest(f"{self.cls} needs messages for weighting them.")
        self._test(weights=self.instance(source=self.source, target=self.target), shape=self.source.shape)


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
        for edge_weights in [None, torch.rand_like(self.source, dtype=torch.get_default_dtype())]:
            y = self.instance(
                x=self.x,
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


class InitializerTestCase(unittest.TestCase):
    """A test case for initializers."""

    #: the number of entities
    num_entities: ClassVar[int] = 33

    #: the shape of the tensor to initialize
    shape: ClassVar[Tuple[int, ...]] = (3,)

    #: to be initialized / set in subclass
    initializer: Initializer

    #: the interaction to use for testing a model
    interaction: ClassVar[HintOrType[Interaction]] = DistMultInteraction
    dtype: ClassVar[torch.dtype] = torch.get_default_dtype()

    def test_initialization(self):
        """Test whether the initializer returns a modified tensor."""
        shape = (self.num_entities, *self.shape)
        if self.dtype.is_complex:
            shape = shape + (2,)
        x = torch.rand(size=shape)
        # initializers *may* work in-place => clone
        y = self.initializer(x.clone())
        assert not (x == y).all()
        self._verify_initialization(y)

    def _verify_initialization(self, x: torch.FloatTensor) -> None:
        """Verify properties of initialization."""
        pass

    def test_model(self):
        """Test whether initializer can be used for a model."""
        triples_factory = generation.generate_triples_factory(num_entities=self.num_entities)
        # actual number may be different...
        self.num_entities = triples_factory.num_entities
        model = pykeen.models.ERModel(
            triples_factory=triples_factory,
            interaction=self.interaction,
            entity_representations_kwargs=dict(shape=self.shape, initializer=self.initializer, dtype=self.dtype),
            relation_representations_kwargs=dict(shape=self.shape),
            random_seed=0,
        ).to(resolve_device())
        model.reset_parameters_()


class PredictBaseTestCase(unittest.TestCase):
    """Base test for prediction workflows."""

    batch_size: ClassVar[int] = 2
    model_cls: ClassVar[Type[Model]]
    model_kwargs: ClassVar[Mapping[str, Any]]

    factory: TriplesFactory
    batch: MappedTriples
    model: Model

    def setUp(self) -> None:
        """Prepare model."""
        self.factory = Nations().training
        self.batch = self.factory.mapped_triples[: self.batch_size, :]
        self.model = self.model_cls(
            triples_factory=self.factory,
            **self.model_kwargs,
        )


class CleanerTestCase(GenericTestCase[Cleaner]):
    """Test cases for cleaner."""

    def post_instantiation_hook(self) -> None:
        """Prepare triples."""
        self.dataset = Nations()
        self.all_entities = set(range(self.dataset.num_entities))
        self.mapped_triples = self.dataset.training.mapped_triples
        # unfavourable split to ensure that cleanup is necessary
        self.reference, self.other = torch.split(
            self.mapped_triples,
            split_size_or_sections=[24, self.mapped_triples.shape[0] - 24],
            dim=0,
        )
        # check for unclean split
        assert get_entities(self.reference) != self.all_entities

    def test_cleanup_pair(self):
        """Test cleanup_pair."""
        reference_clean, other_clean = self.instance.cleanup_pair(
            reference=self.reference,
            other=self.other,
            random_state=42,
        )
        # check that no triple got lost
        assert triple_tensor_to_set(self.mapped_triples) == triple_tensor_to_set(
            torch.cat(
                [
                    reference_clean,
                    other_clean,
                ],
                dim=0,
            )
        )
        # check that triples where only moved from other to reference
        assert is_triple_tensor_subset(self.reference, reference_clean)
        assert is_triple_tensor_subset(other_clean, self.other)
        # check that all entities occur in reference
        assert get_entities(reference_clean) == self.all_entities

    def test_call(self):
        """Test call."""
        triples_groups = [self.reference] + list(torch.split(self.other, split_size_or_sections=3, dim=0))
        clean_groups = self.instance(triples_groups=triples_groups, random_state=42)
        assert all(torch.is_tensor(triples) and triples.dtype for triples in clean_groups)


class SplitterTestCase(GenericTestCase[Splitter]):
    """Test cases for triples splitter."""

    def post_instantiation_hook(self) -> None:
        """Prepare data."""
        dataset = Nations()
        self.all_entities = set(range(dataset.num_entities))
        self.mapped_triples = dataset.training.mapped_triples

    def _test_split(self, ratios: Union[float, Sequence[float]], exp_parts: int):
        """Test splitting."""
        splitted = self.instance.split(
            mapped_triples=self.mapped_triples,
            ratios=ratios,
            random_state=None,
        )
        assert len(splitted) == exp_parts
        # check that no triple got lost
        assert triple_tensor_to_set(self.mapped_triples) == set().union(
            *(triple_tensor_to_set(triples) for triples in splitted)
        )
        # check that all entities are covered in first part
        assert triple_tensor_to_set(splitted[0]) == self.all_entities


class EvaluatorTestCase(unittest_templates.GenericTestCase[Evaluator]):
    """A test case for quickly defining common tests for evaluators models."""

    # the model
    model: Model

    # Settings
    batch_size: int = 8
    embedding_dim: int = 7

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        self.dataset = Nations()
        return super()._pre_instantiation_hook(kwargs=kwargs)

    @property
    def factory(self) -> CoreTriplesFactory:
        """Return the evaluation factory."""
        return self.dataset.validation

    def post_instantiation_hook(self) -> None:  # noqa: D102
        # Use small model (untrained)
        self.model = TransE(triples_factory=self.factory, embedding_dim=self.embedding_dim)

    def _get_input(
        self,
        inverse: bool = False,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Optional[torch.BoolTensor]]:
        # Get batch
        hrt_batch = self.factory.mapped_triples[: self.batch_size].to(self.model.device)

        # Compute scores
        if inverse:
            scores = self.model.score_h(rt_batch=hrt_batch[:, 1:])
        else:
            scores = self.model.score_t(hr_batch=hrt_batch[:, :2])

        # Compute mask only if required
        if self.instance.requires_positive_mask:
            # TODO: Re-use filtering code
            triples = self.factory.mapped_triples.to(self.model.device)
            if inverse:
                sel_col, start_col = 0, 1
            else:
                sel_col, start_col = 2, 0
            stop_col = start_col + 2

            # shape: (batch_size, num_triples)
            triple_mask = (triples[None, :, start_col:stop_col] == hrt_batch[:, None, start_col:stop_col]).all(dim=-1)
            batch_indices, triple_indices = triple_mask.nonzero(as_tuple=True)
            entity_indices = triples[triple_indices, sel_col]

            # shape: (batch_size, num_entities)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[batch_indices, entity_indices] = True
        else:
            mask = None

        return hrt_batch, scores, mask

    def test_process_tail_scores_(self) -> None:
        """Test the evaluator's ``process_tail_scores_()`` function."""
        hrt_batch, scores, mask = self._get_input()
        true_scores = scores[torch.arange(0, hrt_batch.shape[0]), hrt_batch[:, 2]][:, None]
        self.instance.process_scores_(
            hrt_batch=hrt_batch,
            target=LABEL_TAIL,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

    def test_process_head_scores_(self) -> None:
        """Test the evaluator's ``process_head_scores_()`` function."""
        hrt_batch, scores, mask = self._get_input(inverse=True)
        true_scores = scores[torch.arange(0, hrt_batch.shape[0]), hrt_batch[:, 0]][:, None]
        self.instance.process_scores_(
            hrt_batch=hrt_batch,
            target=LABEL_HEAD,
            true_scores=true_scores,
            scores=scores,
            dense_positive_mask=mask,
        )

    def test_finalize(self) -> None:
        """Test the finalize() function."""
        # Process one batch
        hrt_batch, scores, mask = self._get_input()
        true_scores = scores[torch.arange(0, hrt_batch.shape[0]), hrt_batch[:, 2]][:, None]
        for target in (LABEL_HEAD, LABEL_TAIL):
            self.instance.process_scores_(
                hrt_batch=hrt_batch,
                target=target,
                true_scores=true_scores,
                scores=scores,
                dense_positive_mask=mask,
            )

        result = self.instance.finalize()
        assert isinstance(result, MetricResults)

        self._validate_result(
            result=result,
            data={"batch": hrt_batch, "scores": scores, "mask": mask},
        )

    def _validate_result(
        self,
        result: MetricResults,
        data: Dict[str, torch.Tensor],
    ):
        logger.warning(f"{self.__class__.__name__} did not overwrite _validate_result.")

    def test_pipeline(self):
        """Test interaction with pipeline."""
        pipeline(
            training=self.factory,
            testing=self.factory,
            model="distmult",
            evaluator=evaluator_resolver.normalize_cls(self.cls),
            evaluator_kwargs=self.instance_kwargs,
            training_kwargs=dict(
                num_epochs=1,
            ),
        )


class AnchorSelectionTestCase(GenericTestCase[pykeen.nn.node_piece.AnchorSelection]):
    """Tests for anchor selection."""

    num_anchors: int = 7
    num_entities: int = 33
    num_triples: int = 101
    edge_index: numpy.ndarray

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Prepare kwargs."""
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["num_anchors"] = self.num_anchors
        return kwargs

    def post_instantiation_hook(self) -> None:
        """Prepare edge index."""
        generator = numpy.random.default_rng(seed=42)
        self.edge_index = generator.integers(low=0, high=self.num_entities, size=(2, self.num_triples))

    def test_call(self):
        """Test __call__."""
        anchors = self.instance(edge_index=self.edge_index)
        # shape
        assert len(anchors) == self.num_anchors
        # value range
        assert (0 <= anchors).all()
        assert (anchors < self.num_entities).all()
        # no duplicates
        assert len(set(anchors.tolist())) == len(anchors)


class AnchorSearcherTestCase(GenericTestCase[pykeen.nn.node_piece.AnchorSearcher]):
    """Tests for anchor search."""

    num_entities = 33
    k: int = 2
    edge_index: numpy.ndarray
    anchors: numpy.ndarray

    def post_instantiation_hook(self) -> None:
        """Prepare circular edge index."""
        self.edge_index = numpy.stack(
            [numpy.arange(self.num_entities), (numpy.arange(self.num_entities) + 1) % self.num_entities],
            axis=0,
        )
        self.anchors = numpy.arange(0, self.num_entities, 10)

    def test_call(self):
        """Test __call__."""
        tokens = self.instance(edge_index=self.edge_index, anchors=self.anchors, k=self.k)
        # shape
        assert tokens.shape == (self.num_entities, self.k)
        # value range
        assert (tokens >= -1).all()
        assert (tokens < len(self.anchors)).all()
        # no duplicates
        for row in tokens.tolist():
            self.assertDictEqual({k: v for k, v in Counter(row).items() if k >= 0 and v > 1}, {}, msg="duplicate token")


class TokenizerTestCase(GenericTestCase[pykeen.nn.node_piece.Tokenizer]):
    """Tests for tokenization."""

    num_tokens: int = 2
    factory: CoreTriplesFactory

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Prepare triples."""
        self.factory = Nations().training
        return {}

    def test_call(self):
        """Test __call__."""
        vocabulary_size, tokens = self.instance(
            mapped_triples=self.factory.mapped_triples,
            num_tokens=self.num_tokens,
            num_entities=self.factory.num_entities,
            num_relations=self.factory.num_relations,
        )
        assert isinstance(vocabulary_size, int)
        assert vocabulary_size > 0
        # shape
        assert tokens.shape == (self.factory.num_entities, self.num_tokens)
        # value range
        assert (tokens >= -1).all()
        # no repetition, except padding idx
        for row in tokens.tolist():
            self.assertDictEqual({k: v for k, v in Counter(row).items() if k >= 0 and v > 1}, {}, msg="duplicate token")


class NodePieceTestCase(RepresentationTestCase):
    """General test case for node piece representations."""

    cls = pykeen.nn.node_piece.NodePieceRepresentation
    num_entities: ClassVar[int] = 8
    num_relations: ClassVar[int] = 7
    num_triples: ClassVar[int] = 31

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = generation.generate_triples_factory(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_triples=self.num_triples,
            create_inverse_triples=False,
        )
        return kwargs


class EvaluationOnlyModelTestCase(unittest_templates.GenericTestCase[pykeen.models.EvaluationOnlyModel]):
    """Test case for evaluation only models."""

    #: The batch size
    batch_size: int = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        dataset = Nations()
        self.factory = kwargs["triples_factory"] = dataset.training
        return kwargs

    def _verify(self, scores: torch.FloatTensor):
        """Verify scores."""

    def test_score_t(self):
        """Test score_t."""
        hr_batch = self.factory.mapped_triples[torch.randint(self.factory.num_triples, size=(self.batch_size,))][:, :2]
        scores = self.instance.score_t(hr_batch=hr_batch)
        assert scores.shape == (self.batch_size, self.factory.num_entities)
        self._verify(scores)

    def test_score_h(self):
        """Test score_h."""
        rt_batch = self.factory.mapped_triples[torch.randint(self.factory.num_triples, size=(self.batch_size,))][:, 1:]
        scores = self.instance.score_h(rt_batch=rt_batch)
        assert scores.shape == (self.batch_size, self.factory.num_entities)
        self._verify(scores)


class RankBasedMetricTestCase(unittest_templates.GenericTestCase[RankBasedMetric]):
    """A test for rank-based metrics."""

    #: the maximum number of candidates
    max_num_candidates: int = 17

    #: the number of ranks
    num_ranks: int = 33

    #: the number of samples to use for monte-carlo estimation
    num_samples: int = 1_000

    #: the number of candidates for each individual ranking task
    num_candidates: numpy.ndarray

    #: the ranks for each individual ranking task
    ranks: numpy.ndarray

    def post_instantiation_hook(self) -> None:
        """Generate a coherent rank & candidate pair."""
        self.ranks, self.num_candidates = generate_num_candidates_and_ranks(
            num_ranks=self.num_ranks,
            max_num_candidates=self.max_num_candidates,
            seed=42,
        )

    def test_docdata(self):
        """Test the docdata contents of the metric."""
        self.assertTrue(hasattr(self.instance, "increasing"))
        self.assertNotEqual(
            "", self.cls.__doc__.splitlines()[0].strip(), msg="First line of docstring should not be blank"
        )
        self.assertIsNotNone(get_docdata(self.instance), msg="No docdata available")
        self.assertIsNotNone(getattr_or_docdata(self.cls, "link"))
        self.assertIsNotNone(getattr_or_docdata(self.cls, "name"))
        self.assertIsNotNone(getattr_or_docdata(self.cls, "description"))
        self.assertIsNotNone(self.instance.key)

    def _test_call(self, ranks: numpy.ndarray, num_candidates: Optional[numpy.ndarray]):
        """Verify call."""
        x = self.instance(ranks=ranks, num_candidates=num_candidates)
        # data type
        assert isinstance(x, float)
        # value range
        self.assertIn(x, self.instance.value_range.approximate(epsilon=1.0e-08))

    def test_call(self):
        """Test __call__."""
        self._test_call(ranks=self.ranks, num_candidates=self.num_candidates)

    def test_call_best(self):
        """Test __call__ with optimal ranks."""
        self._test_call(ranks=numpy.ones(shape=(self.num_ranks,)), num_candidates=self.num_candidates)

    def test_call_worst(self):
        """Test __call__ with worst ranks."""
        self._test_call(ranks=self.num_candidates, num_candidates=self.num_candidates)

    def test_call_no_candidates(self):
        """Test __call__ without candidates."""
        if self.instance.needs_candidates:
            raise SkipTest(f"{self.instance} requires candidates.")
        self._test_call(ranks=self.ranks, num_candidates=None)

    def test_increasing(self):
        """Test correct increasing annotation."""
        x, y = [
            self.instance(ranks=ranks, num_candidates=self.num_candidates)
            for ranks in [
                # original ranks
                self.ranks,
                # better ranks
                numpy.clip(self.ranks - 1, a_min=1, a_max=None),
            ]
        ]
        if self.instance.increasing:
            self.assertLessEqual(x, y)
        else:
            self.assertLessEqual(y, x)

    def _test_expectation(self, weights: Optional[numpy.ndarray]):
        """Test the numeric expectation is close to the closed form one."""
        try:
            closed = self.instance.expected_value(num_candidates=self.num_candidates, weights=weights)
        except NoClosedFormError as error:
            raise SkipTest("no implementation of closed-form expectation") from error

        generator = numpy.random.default_rng(seed=0)
        low, simulated, high = self.instance.numeric_expected_value_with_ci(
            num_candidates=self.num_candidates,
            num_samples=self.num_samples,
            generator=generator,
            weights=weights,
        )
        self.assertLessEqual(low, closed)
        self.assertLessEqual(closed, high)

    def test_expectation(self):
        """Test the numeric expectation is close to the closed form one."""
        self._test_expectation(weights=None)

    def test_expectation_weighted(self):
        """Test for weighted expectation."""
        self._test_expectation(weights=self._generate_weights())

    def _test_variance(self, weights: Optional[numpy.ndarray]):
        """Test the numeric variance is close to the closed form one."""
        try:
            closed = self.instance.variance(num_candidates=self.num_candidates, weights=weights)
        except NoClosedFormError as error:
            raise SkipTest("no implementation of closed-form variance") from error

        # variances are non-negative
        self.assertLessEqual(0, closed)

        generator = numpy.random.default_rng(seed=0)
        low, simulated, high = self.instance.numeric_variance_with_ci(
            num_candidates=self.num_candidates,
            num_samples=self.num_samples,
            generator=generator,
            weights=weights,
        )
        self.assertLessEqual(low, closed)
        self.assertLessEqual(closed, high)

    def test_variance(self):
        """Test the numeric variance is close to the closed form one."""
        self._test_variance(weights=None)

    def test_variance_weighted(self):
        """Test the weighted numeric variance is close to the closed form one."""
        self._test_variance(weights=self._generate_weights())

    def _generate_weights(self):
        """Generate weights."""
        if not self.instance.supports_weights:
            raise SkipTest(f"{self.instance} does not support weights")
        # generate random weights such that sum = n
        generator = numpy.random.default_rng(seed=21)
        weights = generator.random(size=self.num_candidates.shape)
        weights = self.num_ranks * weights / weights.sum()
        return weights

    def test_different_to_base_metric(self):
        """Check whether the value is different from the base metric (relevant for adjusted metrics)."""
        if not isinstance(self.instance, DerivedRankBasedMetric):
            self.skipTest("no base metric")
        base_instance = rank_based_metric_resolver.make(self.instance.base_cls)
        base_factor = 1 if base_instance.increasing else -1
        self.assertNotEqual(
            self.instance(ranks=self.ranks, num_candidates=self.num_candidates),
            base_factor * base_instance(ranks=self.ranks, num_candidates=self.num_candidates),
        )

    def test_weights_direction(self):
        """Test monotonicity of weighting."""
        if not self.instance.supports_weights:
            raise SkipTest(f"{self.instance} does not support weights")

        # for sanity checking: give the largest weight to best rank => should improve
        idx = self.ranks.argmin()
        weights = numpy.ones_like(self.ranks, dtype=float)
        weights[idx] = 2.0
        weighted = self.instance(ranks=self.ranks, num_candidates=self.num_candidates, weights=weights)
        unweighted = self.instance(ranks=self.ranks, num_candidates=self.num_candidates, weights=None)
        if self.instance.increasing:  # increasing = larger is better => weighted should be better
            self.assertLessEqual(unweighted, weighted)
        else:
            self.assertLessEqual(weighted, unweighted)

    def test_weights_coherence(self):
        """Test coherence for weighted metrics & metric in repeated array."""
        if not self.instance.supports_weights:
            raise SkipTest(f"{self.instance} does not support weights")

        # generate two versions
        generator = numpy.random.default_rng(seed=21)
        repeats = generator.integers(low=1, high=10, size=self.ranks.shape)

        # 1. repeat each rank/candidate pair a random number of times
        repeated_ranks, repeated_num_candidates = [], []
        for rank, num_candidates, repeat in zip(self.ranks, self.num_candidates, repeats):
            repeated_ranks.append(numpy.full(shape=(repeat,), fill_value=rank))
            repeated_num_candidates.append(numpy.full(shape=(repeat,), fill_value=num_candidates))
        repeated_ranks = numpy.concatenate(repeated_ranks)
        repeated_num_candidates = numpy.concatenate(repeated_num_candidates)
        value_repeat = self.instance(ranks=repeated_ranks, num_candidates=repeated_num_candidates, weights=None)

        # 2. do not repeat, but assign a corresponding weight
        weights = repeats.astype(float)
        value_weighted = self.instance(ranks=self.ranks, num_candidates=self.num_candidates, weights=weights)

        self.assertAlmostEqual(value_repeat, value_weighted, delta=2)


class MetricResultTestCase(unittest_templates.GenericTestCase[MetricResults]):
    """Test for metric results."""

    def test_flat_dict(self):
        """Test to_flat_dict."""
        flat_dict = self.instance.to_flat_dict()
        # check flatness
        self.assertIsInstance(flat_dict, dict)
        for key, value in flat_dict.items():
            self.assertIsInstance(key, str)
            # TODO: does this suffice, or do we really need float as datatype?
            self.assertIsInstance(value, (float, int), msg=key)
        self._verify_flat_dict(flat_dict)

    def _verify_flat_dict(self, flat_dict: Mapping[str, Any]):
        pass


class TrainingInstancesTestCase(unittest_templates.GenericTestCase[Instances]):
    """Test for training instances."""

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        self.factory = Nations().training
        return {}

    @abstractmethod
    def _get_expected_length(self) -> int:
        raise NotImplementedError

    def test_getitem(self):
        """Test __getitem__."""
        self.instance: Instances
        assert self.instance[0] is not None

    def test_len(self):
        """Test __len__."""
        self.assertEqual(len(self.instance), self._get_expected_length())

    def test_data_loader(self):
        """Test usage with data loader."""
        for batch in torch.utils.data.DataLoader(
            dataset=self.instance, batch_size=2, shuffle=True, collate_fn=self.instance.get_collator()
        ):
            assert batch is not None


class BatchSLCWATrainingInstancesTestCase(unittest_templates.GenericTestCase[BaseBatchedSLCWAInstances]):
    """Test for batched sLCWA training instances."""

    batch_size: int = 2
    num_negatives_per_positive: int = 3
    kwargs = dict(
        batch_size=batch_size,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=num_negatives_per_positive,
        ),
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        self.factory = Nations().training
        kwargs["mapped_triples"] = self.factory.mapped_triples
        return kwargs

    def test_data_loader(self):
        """Test data loader."""
        for batch in torch.utils.data.DataLoader(dataset=self.instance, batch_size=None):
            assert isinstance(batch, SLCWABatch)
            assert batch.positives.shape == (self.batch_size, 3)
            assert batch.negatives.shape == (self.batch_size, self.num_negatives_per_positive, 3)
            assert batch.masks is None

    def test_length(self):
        """Test length."""
        assert len(self.instance) == len(list(iter(self.instance)))

    def test_data_loader_multiprocessing(self):
        """Test data loader with multiple workers."""
        self.assertEqual(
            sum(
                (
                    batch.positives.shape[0]
                    for batch in torch.utils.data.DataLoader(dataset=self.instance, batch_size=None, num_workers=2)
                )
            ),
            self.factory.num_triples,
        )


class TrainingCallbackTestCase(unittest_templates.GenericTestCase[TrainingCallback]):
    """Base test case for training callbacks."""

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs)
        self.dataset = Nations()
        return kwargs

    def test_pipeline(self):
        """Test running a small pipeline with the provided callback."""
        pipeline(
            dataset=self.dataset,
            model="distmult",
            training_kwargs=dict(
                callbacks=self.instance,
            ),
        )


class GraphPairCombinatorTestCase(unittest_templates.GenericTestCase[GraphPairCombinator]):
    """Base test for graph pair combination methods."""

    def _add_labels(self, tf: CoreTriplesFactory) -> TriplesFactory:
        """Add artificial labels to a triples factory."""
        entity_to_id = {f"e_{i}": i for i in range(tf.num_entities)}
        relation_to_id = {f"r_{i}": i for i in range(tf.num_relations)}
        return TriplesFactory(
            mapped_triples=tf.mapped_triples, entity_to_id=entity_to_id, relation_to_id=relation_to_id
        )

    def _test_combination(self, labels: bool):
        # generate random triples factories
        left, right = [generation.generate_triples_factory(random_state=random_state) for random_state in (0, 1)]
        # generate random alignment
        left_idx, right_idx = torch.stack([torch.arange(left.num_entities), torch.randperm(left.num_entities)])[
            : left.num_entities // 2
        ].numpy()
        # add label information if necessary
        if labels:
            left, right = [self._add_labels(tf) for tf in (left, right)]
            left_idx = [left.entity_id_to_label[i] for i in left_idx]
            right_idx = [right.entity_id_to_label[i] for i in right_idx]
        # prepare alignment data frame
        alignment = pandas.DataFrame(data={EA_SIDE_LEFT: left_idx, EA_SIDE_RIGHT: right_idx})
        # call
        tf_both, alignment_t = self.instance(left=left, right=right, alignment=alignment)
        # check
        assert type(tf_both) is type(left)
        assert alignment_t.ndim == 2
        assert alignment_t.shape[0] == 2
        assert alignment_t.shape[1] <= len(alignment)

    def test_combination_label(self):
        """Test combination with labels."""
        self._test_combination(labels=True)

    def test_combination_id(self):
        """Test combination without labels."""
        self._test_combination(labels=False)

    def test_manual(self):
        """
        Smoke-test on a manual example.

        cf. https://github.com/pykeen/pykeen/pull/893#discussion_r861553903
        """
        left_tf = TriplesFactory.from_labeled_triples(
            pandas.DataFrame(
                [
                    ["la", "0", "lb"],
                    ["lb", "0", "lc"],
                    ["la", "1", "ld"],
                    ["le", "1", "lg"],
                    ["lh", "1", "lg"],
                ],
            ).values
        )
        right_tf = TriplesFactory.from_labeled_triples(
            pandas.DataFrame(
                [
                    ["ra", "2", "rb"],
                    ["ra", "2", "rc"],
                    ["rc", "3", "rd"],
                    ["re", "3", "rg"],
                    ["rh", "3", "rg"],
                ],
            ).values
        )
        test_links = pandas.DataFrame(
            [
                ["ld", "rd"],
                ["le", "re"],
                ["lg", "rg"],
                ["lh", "rh"],
            ],
            columns=[EA_SIDE_LEFT, EA_SIDE_RIGHT],
        )
        combined_tf, alignment_t = self.instance(left=left_tf, right=right_tf, alignment=test_links)
        self._verify_manual(combined_tf=combined_tf)

    @abstractmethod
    def _verify_manual(self, combined_tf: CoreTriplesFactory):
        """Verify the result of the combination of the manual example."""
