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
from typing import Any, ClassVar, Collection, Dict, Generic, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar
from unittest.mock import patch

import pytest
import torch
from click.testing import CliRunner, Result
from torch import optim
from torch.nn import functional
from torch.optim import Adagrad, SGD

import pykeen.models
from pykeen.datasets import Nations
from pykeen.datasets.base import LazyDataset
from pykeen.datasets.kinships import KINSHIPS_TRAIN_PATH
from pykeen.datasets.nations import NATIONS_TEST_PATH, NATIONS_TRAIN_PATH
from pykeen.losses import Loss, PairwiseLoss, PointwiseLoss, SetwiseLoss
from pykeen.models import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model, RESCAL
from pykeen.models.cli import build_cli_from_cls
from pykeen.nn import RepresentationModule
from pykeen.nn.modules import Interaction
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.trackers import ResultTracker
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop, TrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.typing import HeadRepresentation, MappedTriples, RelationRepresentation, TailRepresentation
from pykeen.utils import all_in_bounds, get_subclasses, resolve_device, set_random_seed, unpack_singletons
from tests.constants import EPSILON
from tests.mocks import CustomRepresentations

T = TypeVar("T")

logger = logging.getLogger(__name__)


class GenericTestCase(Generic[T], unittest.TestCase):
    """Generic tests."""

    cls: ClassVar[Type[T]]
    kwargs: ClassVar[Optional[Mapping[str, Any]]] = None
    instance: T
    generator: torch.Generator

    def setUp(self) -> None:
        """Set up the generic testing method."""
        # fix seeds for reproducibility
        _, self.generator, _ = set_random_seed(seed=42)
        kwargs = self.kwargs or {}
        kwargs = self._pre_instantiation_hook(kwargs=dict(kwargs))
        self.instance = self.cls(**kwargs)
        self.post_instantiation_hook()

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Perform actions before instantiation, potentially modyfing kwargs."""
        return kwargs

    def post_instantiation_hook(self) -> None:
        """Perform actions after instantiation."""


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
        try:
            self.dataset._load()
        except (EOFError, IOError):
            self.skipTest('Problem with connection. Try this test again later.')

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


# TODO update
class LossTestCase(GenericTestCase[Loss]):
    """Base unittest for loss functions."""

    #: The batch size
    batch_size: ClassVar[int] = 3

    def _check_loss_value(self, loss_value: torch.FloatTensor) -> None:
        """Check loss value dimensionality, and ability for backward."""
        # test reduction
        self.assertEqual(0, loss_value.ndim)

        # test finite loss value
        self.assertTrue(torch.isfinite(loss_value))

        # Test backward
        loss_value.backward()


# TODO update
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


# TODO update
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


# TODO update
class SetwiseLossTestCase(LossTestCase):
    """Unit tests for setwise losses."""

    #: The number of entities.
    num_entities: int = 13

    def test_type(self):
        """Test the loss is the right type."""
        self.assertIsInstance(self.instance, SetwiseLoss)

    def test_forward(self):
        """Test forward(scores, labels)."""
        scores = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        labels = torch.rand(self.batch_size, self.num_entities, requires_grad=False)
        loss_value = self.instance(
            scores,
            labels,
        )
        self._check_loss_value(loss_value=loss_value)


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
        self.shape_kwargs.setdefault("d", self.dim)
        result = tuple(
            tuple(
                torch.rand(*prefix_shape, *(self.shape_kwargs[dim] for dim in weight_shape), requires_grad=True)
                for weight_shape in weight_shapes
            )
            for prefix_shape, weight_shapes in zip(
                shapes,
                [self.cls.entity_shape, self.cls.relation_shape, self.cls.entity_shape],
            )
        )
        return tuple(unpack_singletons(*result))

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

    def test_score_hrt(self):
        """Test score_hrt."""
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.batch_size,),
        )
        scores = self.instance.score_hrt(h=h, r=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, 1))

    def test_score_h(self):
        """Test score_h."""
        h, r, t = self._get_hrt(
            (self.num_entities,),
            (self.batch_size,),
            (self.batch_size,),
        )
        scores = self.instance.score_h(all_entities=h, r=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

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
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.num_relations,),
            (self.batch_size,),
        )
        scores = self.instance.score_r(h=h, all_relations=r, t=t)
        if len(self.cls.relation_shape) == 0:
            exp_shape = (self.batch_size, 1)
        else:
            exp_shape = (self.batch_size, self.num_relations)
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
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.num_entities,),
        )
        scores = self.instance.score_t(h=h, r=r, all_entities=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

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
            kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)

            # calculate by functional
            scores_f = self.cls.func(**kwargs).view(-1)

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


class TestsTestCase(Generic[T], unittest.TestCase):
    """A generic test for tests."""

    base_cls: Type[T]
    base_test: Type[GenericTestCase[T]]
    skip_cls: Collection[T] = tuple()

    def test_testing(self):
        """Check that there is a test for all subclasses."""
        to_test = set(get_subclasses(self.base_cls)).difference(self.skip_cls)
        tested = (test_cls.cls for test_cls in get_subclasses(self.base_test) if hasattr(test_cls, "cls"))
        not_tested = to_test.difference(tested)
        assert not not_tested, not_tested


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
        a = torch.rand(self.batch_size, 10, device=self.device, generator=self.generator)
        b = torch.rand(self.batch_size, 20, device=self.device, generator=self.generator)

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
        x = torch.rand(self.batch_size, 10, generator=self.generator)

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

    def _expected_penalty(self, x: torch.FloatTensor) -> torch.FloatTensor:
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


class ModelTestCase(unittest.TestCase):
    """A test case for quickly defining common tests for KGE models."""

    #: The class of the model to test
    model_cls: ClassVar[Type[Model]]

    #: Additional arguments passed to the model's constructor method
    model_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    #: Additional arguments passed to the training loop's constructor method
    training_loop_kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    #: The triples factory instance
    factory: TriplesFactory

    #: The model instance
    model: Model

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

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        _, self.generator, _ = set_random_seed(42)

        dataset = Nations(create_inverse_triples=self.create_inverse_triples)
        self.factory = dataset.training
        self.model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            **(self.model_kwargs or {}),
        ).to_device_()

    def test_get_grad_parameters(self):
        """Test the model's ``get_grad_params()`` method."""
        # assert there is at least one trainable parameter
        assert len(list(self.model.get_grad_params())) > 0

        # Check that all the parameters actually require a gradient
        for parameter in self.model.get_grad_params():
            assert parameter.requires_grad

        # Try to initialize an optimizer
        optimizer = SGD(params=self.model.get_grad_params(), lr=1.0)
        assert optimizer is not None

    def test_reset_parameters_(self):
        """Test :func:`Model.reset_parameters_`."""
        # get model parameters
        params = list(self.model.parameters())
        old_content = {
            id(p): p.data.detach().clone()
            for p in params
        }

        # re-initialize
        self.model.reset_parameters_()

        # check that the operation works in-place
        new_params = list(self.model.parameters())
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
            torch.save(self.model, os.path.join(temp_directory, 'model.pickle'))

    def test_score_hrt(self) -> None:
        """Test the model's ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.model.device)
        try:
            scores = self.model.score_hrt(batch)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, 1)
        self._check_scores(batch, scores)

    def test_score_t(self) -> None:
        """Test the model's ``score_t()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, :2].to(self.model.device)
        # assert batch comprises (head, relation) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores = self.model.score_t(batch)
        except NotImplementedError:
            self.fail(msg='Score_o not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    def test_score_h(self) -> None:
        """Test the model's ``score_h()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.model.device)
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores = self.model.score_h(batch)
        except NotImplementedError:
            self.fail(msg='Score_s not yet implemented')
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        assert scores.shape == (self.batch_size, self.model.num_entities)
        self._check_scores(batch, scores)

    @pytest.mark.slow
    def test_train_slcwa(self) -> None:
        """Test that sLCWA training does not fail."""
        loop = SLCWATrainingLoop(
            model=self.model,
            optimizer=Adagrad(params=self.model.get_grad_params(), lr=0.001),
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
            model=self.model,
            optimizer=Adagrad(params=self.model.get_grad_params(), lr=0.001),
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
            losses = loop.train(num_epochs=num_epochs, batch_size=batch_size, sampler=sampler, use_tqdm=False)
        except RuntimeError as e:
            if str(e) == 'fft: ATen not compiled with MKL support':
                self.skipTest(str(e))
            else:
                raise e
        else:
            return losses

    def test_save_load_model_state(self):
        """Test whether a saved model state can be re-loaded."""
        original_model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            random_seed=42,
            **(self.model_kwargs or {}),
        ).to_device_()

        loaded_model = self.model_cls(
            self.factory,
            embedding_dim=self.embedding_dim,
            random_seed=21,
            **(self.model_kwargs or {}),
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
    def cli_extras(self):
        """Return a list of extra flags for the CLI."""
        kwargs = self.model_kwargs or {}
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
        extras = [str(e) for e in extras]
        return extras

    @pytest.mark.slow
    def test_cli_training_nations(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH] + self.cli_extras)

    @pytest.mark.slow
    def test_cli_training_kinships(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', KINSHIPS_TRAIN_PATH] + self.cli_extras)

    @pytest.mark.slow
    def test_cli_training_nations_testing(self):
        """Test running the pipeline on almost all models with only training data."""
        self._help_test_cli(['-t', NATIONS_TRAIN_PATH, '-q', NATIONS_TEST_PATH] + self.cli_extras)

    def _help_test_cli(self, args):
        """Test running the pipeline on all models."""
        if issubclass(self.model_cls, pykeen.models.RGCN):
            self.skipTest('There is a problem with non-reproducible unittest for R-GCN.')
        runner = CliRunner()
        cli = build_cli_from_cls(self.model_cls)
        # TODO: Catch HolE MKL error?
        result: Result = runner.invoke(cli, args)

        self.assertEqual(
            0,
            result.exit_code,
            msg=f'''
Command
=======
$ pykeen train {self.model_cls.__name__.lower()} {' '.join(args)}

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
            d = self.model_cls.hpo_default
        except AttributeError:
            self.fail(msg=f'{self.model_cls.__name__} is missing hpo_default class attribute')
        else:
            self.assertIsInstance(d, dict)

    def test_post_parameter_update_regularizer(self):
        """Test whether post_parameter_update resets the regularization term."""
        # set regularizer term
        self.model.regularizer.regularization_term = None

        # call post_parameter_update
        self.model.post_parameter_update()

        # assert that the regularization term has been reset
        assert self.model.regularizer.regularization_term == torch.zeros(1, dtype=torch.float, device=self.model.device)

    def test_post_parameter_update(self):
        """Test whether post_parameter_update correctly enforces model constraints."""
        # do one optimization step
        opt = optim.SGD(params=self.model.parameters(), lr=1.)
        batch = self.factory.mapped_triples[:self.batch_size, :].to(self.model.device)
        scores = self.model.score_hrt(hrt_batch=batch)
        fake_loss = scores.mean()
        fake_loss.backward()
        opt.step()

        # call post_parameter_update
        self.model.post_parameter_update()

        # check model constraints
        self._check_constraints()

    def _check_constraints(self):
        """Check model constraints."""

    def test_score_h_with_score_hrt_equality(self) -> None:
        """Test the equality of the model's  ``score_h()`` and ``score_hrt()`` function."""
        batch = self.factory.mapped_triples[:self.batch_size, 1:].to(self.model.device)
        self.model.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_relations).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_h = self.model.score_h(batch)
            scores_hrt = super(self.model.__class__, self.model).score_h(batch)
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
        batch = self.factory.mapped_triples[:self.batch_size, [0, 2]].to(self.model.device)
        self.model.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_entities).all()
        try:
            scores_r = self.model.score_r(batch)
            scores_hrt = super(self.model.__class__, self.model).score_r(batch)
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
        batch = self.factory.mapped_triples[:self.batch_size, :-1].to(self.model.device)
        self.model.eval()
        # assert batch comprises (relation, tail) pairs
        assert batch.shape == (self.batch_size, 2)
        assert (batch[:, 0] < self.factory.num_entities).all()
        assert (batch[:, 1] < self.factory.num_relations).all()
        try:
            scores_t = self.model.score_t(batch)
            scores_hrt = super(self.model.__class__, self.model).score_t(batch)
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
        with patch.object(self.model_cls, 'reset_parameters_', return_value=None) as mock_method:
            try:
                self.model_cls(
                    triples_factory=self.factory,
                    embedding_dim=self.embedding_dim,
                    **(self.model_kwargs or {}),
                )
            except TypeError as error:
                assert error.args == ("'NoneType' object is not callable",)
            mock_method.assert_called_once()

    def test_custom_representations(self):
        """Tests whether we can provide custom representations."""
        if isinstance(self.model, EntityEmbeddingModel):
            old_embeddings = self.model.entity_embeddings
            self.model.entity_embeddings = CustomRepresentations(
                num_entities=self.factory.num_entities,
                embedding_dim=old_embeddings.embedding_dim,
            )
            # call some functions
            self.model.reset_parameters_()
            self.test_score_hrt()
            self.test_score_t()
            # reset to old state
            self.model.entity_embeddings = old_embeddings
        elif isinstance(self.model, EntityRelationEmbeddingModel):
            old_embeddings = self.model.relation_embeddings
            self.model.relation_embeddings = CustomRepresentations(
                num_entities=self.factory.num_relations,
                embedding_dim=old_embeddings.embedding_dim,
            )
            # call some functions
            self.model.reset_parameters_()
            self.test_score_hrt()
            self.test_score_t()
            # reset to old state
            self.model.relation_embeddings = old_embeddings
        else:
            self.skipTest(f'Not testing custom representations for model: {self.model.__class__.__name__}')


class DistanceModelTestCase(ModelTestCase):
    """A test case for distance-based models."""

    def _check_scores(self, batch, scores) -> None:
        super()._check_scores(batch=batch, scores=scores)
        # Distance-based model
        assert (scores <= 0.0).all()


class BaseKG2ETest(ModelTestCase):
    """General tests for the KG2E model."""

    model_cls = pykeen.models.KG2E

    def _check_constraints(self):
        """Check model constraints.

        * Entity and relation embeddings have to have at most unit L2 norm.
        * Covariances have to have values between c_min and c_max
        """
        for embedding in (self.model.entity_embeddings, self.model.relation_embeddings):
            assert all_in_bounds(embedding(indices=None).norm(p=2, dim=-1), high=1., a_tol=EPSILON)
        for cov in (self.model.entity_covariances, self.model.relation_covariances):
            assert all_in_bounds(cov(indices=None), low=self.model.c_min, high=self.model.c_max)


class BaseNTNTest(ModelTestCase):
    """Test the NTN model."""

    model_cls = pykeen.models.NTN

    def test_can_slice(self):
        """Test that the slicing properties are calculated correctly."""
        self.assertTrue(self.model.can_slice_h)
        self.assertFalse(self.model.can_slice_r)
        self.assertTrue(self.model.can_slice_t)


class BaseRGCNTest(ModelTestCase):
    """Test the R-GCN model."""

    model_cls = pykeen.models.RGCN
    sampler = 'schlichtkrull'

    def _check_constraints(self):
        """Check model constraints.

        Enriched embeddings have to be reset.
        """
        assert self.model.entity_representations.enriched_embeddings is None
