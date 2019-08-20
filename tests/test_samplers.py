# -*- coding: utf-8 -*-

"""Test that samplers can be executed."""

import unittest
from typing import ClassVar, Type

import numpy
import torch

from poem.datasets import NationsTrainingTriplesFactory
from poem.instance_creation_factories import OWAInstances, TriplesFactory
from poem.negative_sampling import BasicNegativeSampler, NegativeSampler


def _array_check_bounds(
    array: torch.LongTensor,
    low: int,
    high: int,
) -> bool:
    """Check if all elements lie in bounds."""
    return (low <= array).all() and (array < high).all()


class _NegativeSamplingTestCase:
    """A test case for quickly defining common tests for samplers."""

    #: The batch size
    batch_size: int
    #: The random seed
    seed: int
    #: The triples factory
    triples_factory: TriplesFactory
    #: The OWA instances
    owa_instances: OWAInstances
    #: Class of negative sampling to test
    negative_sampling_cls: ClassVar[Type[NegativeSampler]]
    #: The negative sampler instance, initialized in setUp
    negative_sampler: NegativeSampler
    #: A positive batch
    positive_batch: torch.LongTensor

    def setUp(self) -> None:
        """Set up the test case with a triples factory and model."""
        self.batch_size = 16
        self.seed = 42
        self.triples_factory = NationsTrainingTriplesFactory()
        self.owa_instances = self.triples_factory.create_owa_instances()
        self.negative_sampler = self.negative_sampling_cls(triples_factory=self.triples_factory)
        random = numpy.random.RandomState(seed=self.seed)
        batch_indices = random.randint(low=0, high=self.owa_instances.num_instances, size=(self.batch_size,))
        self.positive_batch = torch.tensor(self.owa_instances.mapped_triples[batch_indices], dtype=torch.long)

    def test_sample(self) -> None:
        # Generate negative sample
        negative_batch = self.negative_sampler.sample(positive_batch=self.positive_batch)

        # check shape
        assert negative_batch.shape == self.positive_batch.shape

        # check bounds: subjects
        assert _array_check_bounds(negative_batch[:, 0], low=0, high=self.triples_factory.num_entities)

        # check bounds: relations
        assert _array_check_bounds(negative_batch[:, 1], low=0, high=self.triples_factory.num_relations)

        # check bounds: objects
        assert _array_check_bounds(negative_batch[:, 2], low=0, high=self.triples_factory.num_entities)

        # Assert arrays not equal
        assert not (negative_batch != self.positive_batch).all()


class BasicNegativeSamplerTest(_NegativeSamplingTestCase, unittest.TestCase):
    """Test the basic negative sampler."""

    negative_sampling_cls = BasicNegativeSampler
