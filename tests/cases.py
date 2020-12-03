# -*- coding: utf-8 -*-

"""Test cases for PyKEEN."""

import tempfile
import timeit
import unittest
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from torch.nn import functional

from pykeen.datasets.base import LazyDataset
from pykeen.losses import Loss, PairwiseLoss, PointwiseLoss, SetwiseLoss
from pykeen.triples import TriplesFactory


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


class LossTestCase(unittest.TestCase):
    """Base unittest for loss functions."""

    #: The class
    cls: ClassVar[Type[Loss]]

    #: Constructor keyword arguments
    kwargs: ClassVar[Optional[Mapping[str, Any]]] = None

    #: The loss instance
    loss: Loss

    #: The batch size
    batch_size: ClassVar[int] = 3

    def setUp(self) -> None:
        """Initialize the instance."""
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        self.loss = self.cls(**kwargs)

    def _check_loss_value(self, loss_value: torch.FloatTensor) -> None:
        """Check loss value dimensionality, and ability for backward."""
        # test reduction
        self.assertEqual(0, loss_value.ndim)

        # Test backward
        loss_value.backward()


class PointwiseLossTestCase(LossTestCase):
    """Base unit test for label-based losses."""

    #: The number of entities.
    num_entities: int = 17

    def test_type(self):
        """Test the loss is the right type."""
        self.assertIsInstance(self.loss, PointwiseLoss)

    def test_label_loss(self):
        """Test ``forward(logits, labels)``."""
        logits = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        labels = functional.normalize(torch.rand(self.batch_size, self.num_entities, requires_grad=False), p=1, dim=-1)
        loss_value = self.loss.forward(
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
        self.assertIsInstance(self.loss, PairwiseLoss)

    def test_pair_loss(self):
        """Test ``forward(pos_scores, neg_scores)``."""
        pos_scores = torch.rand(self.batch_size, 1, requires_grad=True)
        neg_scores = torch.rand(self.batch_size, self.num_negatives, requires_grad=True)
        loss_value = self.loss.forward(
            pos_scores,
            neg_scores,
        )
        self._check_loss_value(loss_value)


class SetwiseLossTestCase(LossTestCase):
    """unittests for setwise losses."""

    #: Setwise do not support slcwa training loop
    training_loop_support = dict(
        slcwa=False,
        lcwa=True,
    )

    #: The number of entities.
    num_entities: int = 13

    def test_type(self):
        """Test the loss is the right type."""
        self.assertIsInstance(self.loss, SetwiseLoss)

    def test_forward(self):
        """Test forward(scores, labels)."""
        scores = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        labels = torch.rand(self.batch_size, self.num_entities, requires_grad=False)
        loss_value = self.loss.forward(
            scores,
            labels,
        )
        self._check_loss_value(loss_value=loss_value)
