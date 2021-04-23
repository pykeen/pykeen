# -*- coding: utf-8 -*-

"""Tests for filterers."""

from typing import Any, MutableMapping

import numpy
import unittest_templates

from pykeen.datasets import Nations
from pykeen.sampling.filtering import BloomFilterer, DefaultFilterer, Filterer


class FiltererTest(unittest_templates.GenericTestCase[Filterer]):
    """A basic test for filtering."""

    seed = 42
    batch_size = 16
    num_negs_per_pos = 10

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["triples_factory"] = self.triples_factory = Nations().training
        return kwargs

    def post_instantiation_hook(self) -> None:  # noqa: D102
        seed = 42
        random = numpy.random.RandomState(seed=seed)
        self.slcwa_instances = self.triples_factory.create_slcwa_instances()
        batch_indices = random.randint(low=0, high=len(self.slcwa_instances), size=(self.batch_size,))
        self.positive_batch = self.slcwa_instances.mapped_triples[batch_indices]

    def test_filter(self):
        """Test the filter method."""
        # Check whether filtering works correctly
        # First giving an example where all triples have to be filtered
        _, batch_filter = self.instance(negative_batch=self.positive_batch)
        # The filter should remove all triples
        assert batch_filter.sum() == 0
        # Create an example where no triples will be filtered
        _, batch_filter = self.instance(
            negative_batch=(self.positive_batch + self.triples_factory.num_entities),
        )
        # The filter should not remove any triple
        assert self.positive_batch.size()[0] == batch_filter.sum()


class DefaultFiltererTests(FiltererTest):
    """Tests for the default filterer."""

    cls = DefaultFilterer


class BloomFiltererTest(FiltererTest):
    """Tests for the bloom filterer."""

    cls = BloomFilterer


class FiltererMetaTestCase(unittest_templates.MetaTestCase[Filterer]):
    """Test all filterers are tested."""

    base_cls = Filterer
    base_test = FiltererTest
