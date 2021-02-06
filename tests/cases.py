# -*- coding: utf-8 -*-

"""Test cases for PyKEEN."""

import logging
import pathlib
import tempfile
import timeit
import unittest
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Collection, Generic, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar

import torch
from torch.nn import functional

from pykeen.datasets.base import LazyDataset
from pykeen.losses import Loss, PairwiseLoss, PointwiseLoss, SetwiseLoss
from pykeen.nn.modules import Interaction
from pykeen.trackers import ResultTracker
from pykeen.triples import TriplesFactory
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from pykeen.utils import get_subclasses, set_random_seed, unpack_singletons

T = TypeVar("T")

logger = logging.getLogger(__name__)


class GenericTestCase(Generic[T], unittest.TestCase):
    """Generic tests."""

    cls: Type[T]
    kwargs: Optional[Mapping[str, Any]] = None
    instance: T

    def setUp(self) -> None:
        """Set up the generic testing method."""
        # fix seeds for reproducibility
        set_random_seed(seed=42)
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
