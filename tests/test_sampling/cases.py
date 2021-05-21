# -*- coding: utf-8 -*-

"""Test cases for sampling."""

from typing import Any, MutableMapping

import numpy
import torch
import unittest_templates

from pykeen.datasets import Nations
from pykeen.sampling import NegativeSampler
from pykeen.sampling.filtering import BloomFilterer, PythonSetFilterer
from pykeen.triples import Instances, TriplesFactory

__all__ = [
    'NegativeSamplerGenericTestCase',
]


def _array_check_bounds(
    array: torch.LongTensor,
    low: int,
    high: int,
) -> bool:
    """Check if all elements lie in bounds."""
    return (low <= array).all() and (array < high).all()


class NegativeSamplerGenericTestCase(unittest_templates.GenericTestCase[NegativeSampler]):
    """A test case for quickly defining common tests for samplers."""

    #: The batch size
    batch_size: int = 16
    #: The random seed
    seed: int = 42
    #: The triples factory
    triples_factory: TriplesFactory
    #: The instances
    training_instances: Instances
    #: A positive batch
    positive_batch: torch.LongTensor
    #: Kwargs
    kwargs = {
        'num_negs_per_pos': 10,
    }

    def pre_setup_hook(self) -> None:
        """Set up the test case with a triples factory, training instances, and a default positive batch."""
        self.triples_factory = Nations().training
        self.training_instances = self.triples_factory.create_slcwa_instances()
        random_state = numpy.random.RandomState(seed=self.seed)
        batch_indices = random_state.randint(low=0, high=len(self.training_instances), size=(self.batch_size,))
        self.positive_batch = self.training_instances.mapped_triples[batch_indices]

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs['triples_factory'] = self.triples_factory
        return kwargs

    def check_sample(self, instance: NegativeSampler) -> None:
        """Test generating a negative sample."""
        # Generate negative sample
        negative_batch, batch_filter = instance.sample(positive_batch=self.positive_batch)

        # check filter shape if necessary
        if instance.filterer is not None:
            assert batch_filter is not None
            assert batch_filter.shape == (self.batch_size, instance.num_negs_per_pos)
            assert batch_filter.dtype == torch.bool
        else:
            assert batch_filter is None

        # check shape
        assert negative_batch.shape == (self.positive_batch.shape[0], instance.num_negs_per_pos, 3)

        # check bounds: heads
        assert _array_check_bounds(negative_batch[..., 0], low=0, high=self.triples_factory.num_entities)

        # check bounds: relations
        assert _array_check_bounds(negative_batch[..., 1], low=0, high=self.triples_factory.num_relations)

        # check bounds: tails
        assert _array_check_bounds(negative_batch[..., 2], low=0, high=self.triples_factory.num_entities)

        if instance.filterer is not None:
            positive_batch = self.positive_batch.unsqueeze(dim=1).repeat(1, instance.num_negs_per_pos, 1)
            positive_batch = positive_batch[batch_filter]
            negative_batch = negative_batch[batch_filter]

            # test that the negative triple is not the original positive triple
            assert (negative_batch != positive_batch).any(dim=-1).all()

    def test_sample_no_filter(self) -> None:
        """Test generating a negative sample."""
        self.check_sample(self.instance)

    def test_sample_set_filtered(self) -> None:
        """Test generating a negative sample with set-based filtering."""
        instance = self.cls(**self.instance_kwargs, filterer=PythonSetFilterer)
        self.check_sample(instance)

    def test_sample_bloom_filtered(self):
        """Test generating a negative sample with bloom filtering."""
        instance = self.cls(**self.instance_kwargs, filterer=BloomFilterer)
        self.check_sample(instance)

    def _update_positive_batch(self, positive_batch, batch_filter):
        # shape: (batch_size, 1, num_neg)
        positive_batch = positive_batch.unsqueeze(dim=1)

        if batch_filter is not None:
            positive_batch = positive_batch[batch_filter]
        return positive_batch

    def test_small_batch(self):
        """Test on a small batch."""
        self.instance.sample(positive_batch=self.positive_batch[:1])
