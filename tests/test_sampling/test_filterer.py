# -*- coding: utf-8 -*-

"""Tests for filterers."""

from typing import Any, MutableMapping

import torch
import unittest_templates

from pykeen.datasets import Nations
from pykeen.sampling.filtering import BloomFilterer, DefaultFilterer, Filterer, PythonSetFilterer
from pykeen.utils import set_random_seed


class FiltererTest(unittest_templates.GenericTestCase[Filterer]):
    """A basic test for filtering."""

    seed = 42
    batch_size = 16
    num_negs_per_pos = 10

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        self.generator = set_random_seed(seed=self.seed)[1]
        kwargs["triples_factory"] = self.triples_factory = Nations().training
        return kwargs

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.slcwa_instances = self.triples_factory.create_slcwa_instances()
        self.positive_batch = self.slcwa_instances.mapped_triples[torch.randint(
            low=0,
            high=len(self.slcwa_instances),
            size=(self.batch_size,),
            generator=self.generator,
        )]

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


class PythonSetFiltererTest(FiltererTest):
    """Tests for the Python set-based filterer."""

    cls = PythonSetFilterer


class BloomFiltererTest(FiltererTest):
    """Tests for the bloom filterer."""

    cls = BloomFilterer


class FiltererMetaTestCase(unittest_templates.MetaTestCase[Filterer]):
    """Test all filterers are tested."""

    base_cls = Filterer
    base_test = FiltererTest
