"""Tests for filterers."""

import warnings
from collections.abc import MutableMapping
from typing import Any

import pytest
import torch
import unittest_templates

from pykeen.datasets import Nations
from pykeen.sampling import BasicNegativeSampler
from pykeen.sampling.filtering import (
    BloomFilterer,
    Filterer,
    PythonSetFilterer,
    SortedKeyFilterer,
    SortedKeyOverflowError,
    filterer_resolver,
    make_default_filterer,
)
from pykeen.utils import set_random_seed


class FiltererTest(unittest_templates.GenericTestCase[Filterer]):
    """A basic test for filtering."""

    seed = 42
    batch_size = 16
    num_negs_per_pos = 10

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        self.generator = set_random_seed(seed=self.seed)[1]
        self.triples_factory = Nations().training
        kwargs["mapped_triples"] = self.mapped_triples = self.triples_factory.mapped_triples
        return kwargs

    def post_instantiation_hook(self) -> None:
        self.positive_batch = self.mapped_triples[
            torch.randint(
                low=0,
                high=self.mapped_triples.shape[0],
                size=(self.batch_size,),
                generator=self.generator,
            )
        ]

    def test_filter(self):
        """Test the filter method."""
        # Check whether filtering works correctly
        # First giving an example where all triples have to be filtered
        batch_filter = self.instance(negative_batch=self.positive_batch)
        # The filter should remove all triples
        assert batch_filter.sum() == 0
        # Create an example where no triples will be filtered
        batch_filter = self.instance(
            negative_batch=(self.positive_batch + self.triples_factory.num_entities),
        )
        # The filter should not remove any triple
        assert self.positive_batch.size()[0] == batch_filter.sum()


class PythonSetFiltererTest(FiltererTest):
    """Tests for the Python set-based filterer."""

    cls = PythonSetFilterer


class BloomFiltererTest(FiltererTest):
    """Tests for the bloom filterer."""

    cls = BloomFilterer


class SortedKeyFiltererTest(FiltererTest):
    """Tests for the sorted key filterer."""

    cls = SortedKeyFilterer

    def test_consistent_with_python_set(self):
        """Test that results match the exact Python set-based filterer, incl. out-of-range IDs and batch shapes."""
        generator = torch.Generator().manual_seed(self.seed)
        num_entities, num_relations = 50, 7
        mapped_triples = torch.stack(
            [
                torch.randint(num_entities, size=(500,), generator=generator),
                torch.randint(num_relations, size=(500,), generator=generator),
                torch.randint(num_entities, size=(500,), generator=generator),
            ],
            dim=-1,
        )
        filterer = SortedKeyFilterer(mapped_triples=mapped_triples)
        reference = PythonSetFilterer(mapped_triples=mapped_triples)
        # query IDs partially exceed the maximum stored IDs, and include negative IDs
        negatives = torch.stack(
            [
                torch.randint(-2, num_entities + 5, size=(8, 300), generator=generator),
                torch.randint(-2, num_relations + 5, size=(8, 300), generator=generator),
                torch.randint(-2, num_entities + 5, size=(8, 300), generator=generator),
            ],
            dim=-1,
        )
        # mix in some known positives
        negatives[:, :20] = mapped_triples[torch.randint(500, size=(8, 20), generator=generator)]
        expected = reference.contains(batch=negatives)
        result = filterer.contains(batch=negatives)
        assert result.shape == negatives.shape[:-1]
        assert result.any()
        assert torch.equal(result, expected)
        # forward
        assert torch.equal(filterer(negative_batch=negatives), ~expected)

    def test_no_collision(self):
        """Test that IDs exceeding the stored ranges do not collide with stored keys."""
        # stored: (0, 0, 2) has key 2 with n_t = 3; (0, 0, 2) and (0, 1, -1) must not be confused
        filterer = SortedKeyFilterer(mapped_triples=torch.as_tensor([[0, 0, 2], [1, 1, 0]]))
        queries = torch.as_tensor([[0, 0, 2], [0, 1, -1], [0, 0, 5], [0, 2, 0], [1, 1, 0]])
        assert filterer.contains(batch=queries).tolist() == [True, False, False, False, True]

    def test_empty(self):
        """Test filtering with no stored triples."""
        filterer = SortedKeyFilterer(mapped_triples=torch.empty(0, 3, dtype=torch.long))
        assert not filterer.contains(batch=self.positive_batch).any()

    def test_explicit_sizes(self):
        """Test explicitly passing sizes."""
        filterer = SortedKeyFilterer(
            mapped_triples=self.mapped_triples,
            num_entities=self.triples_factory.num_entities + 3,
            num_relations=self.triples_factory.num_relations + 3,
        )
        assert filterer.contains(batch=self.positive_batch).all()
        with pytest.raises(ValueError, match="smaller than the observed size"):
            SortedKeyFilterer(mapped_triples=self.mapped_triples, num_entities=1)

    def test_overflow(self):
        """Test that an error is raised if keys would overflow."""
        with pytest.raises(SortedKeyOverflowError, match="overflow"):
            SortedKeyFilterer(mapped_triples=torch.as_tensor([[2**22, 2**20, 2**22]]))


def test_default_filterer():
    """Test that the sorted key filterer is the default filterer."""
    mapped_triples = Nations().training.mapped_triples
    assert filterer_resolver.default is SortedKeyFilterer
    assert isinstance(make_default_filterer(mapped_triples=mapped_triples), SortedKeyFilterer)
    sampler = BasicNegativeSampler(mapped_triples=mapped_triples, filtered=True)
    assert isinstance(sampler.filterer, SortedKeyFilterer)


def test_default_filterer_fallback(caplog: pytest.LogCaptureFixture):
    """Test that the default filterer falls back to the bloom filterer if the sorted keys would overflow."""
    mapped_triples = Nations().training.mapped_triples
    # sizes are only used for the overflow check, i.e., no large tensors are allocated
    huge = {"num_entities": 2**22, "num_relations": 2**20}
    with caplog.at_level("WARNING", logger="pykeen.sampling.filtering"):
        filterer = make_default_filterer(mapped_triples=mapped_triples, **huge)
    assert isinstance(filterer, BloomFilterer)
    assert "Falling back" in caplog.text
    # also via the negative sampler, with inferred sizes
    sampler = BasicNegativeSampler(
        mapped_triples=torch.as_tensor([[2**22, 2**20, 2**22]]), num_negs_per_pos=1, filtered=True
    )
    assert isinstance(sampler.filterer, BloomFilterer)


def test_filterer_kwargs_without_filterer_deprecated():
    """Test that filterer kwargs without an explicit filterer configure the bloom filterer, with a deprecation."""
    mapped_triples = Nations().training.mapped_triples
    with pytest.warns(DeprecationWarning, match="filterer='bloom'"):
        sampler = BasicNegativeSampler(
            mapped_triples=mapped_triples, filtered=True, filterer_kwargs={"error_rate": 0.0001}
        )
    assert isinstance(sampler.filterer, BloomFilterer)
    assert sampler.filterer.error_rate == 0.0001
    # empty kwargs use the default
    sampler = BasicNegativeSampler(mapped_triples=mapped_triples, filtered=True, filterer_kwargs={})
    assert isinstance(sampler.filterer, SortedKeyFilterer)


def test_explicit_filterer_no_fallback():
    """Test that explicitly requested filterers are used, and the sorted key filterer does not fall back."""
    mapped_triples = Nations().training.mapped_triples
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sampler = BasicNegativeSampler(mapped_triples=mapped_triples, filtered=True, filterer="bloom")
        assert isinstance(sampler.filterer, BloomFilterer)
        sampler = BasicNegativeSampler(
            mapped_triples=mapped_triples, filtered=True, filterer="bloom", filterer_kwargs={"error_rate": 0.0001}
        )
        assert isinstance(sampler.filterer, BloomFilterer)
        assert sampler.filterer.error_rate == 0.0001
        sampler = BasicNegativeSampler(
            mapped_triples=mapped_triples,
            filtered=True,
            filterer="sorted-key",
            filterer_kwargs={"num_entities": 100, "num_relations": 100},
        )
        assert isinstance(sampler.filterer, SortedKeyFilterer)
        assert sampler.filterer.sizes.tolist() == [100, 100, 100]
    with pytest.raises(SortedKeyOverflowError):
        BasicNegativeSampler(
            mapped_triples=mapped_triples,
            filtered=True,
            filterer="sorted-key",
            filterer_kwargs={"num_entities": 2**22, "num_relations": 2**20},
        )


class FiltererMetaTestCase(unittest_templates.MetaTestCase[Filterer]):
    """Test all filterers are tested."""

    base_cls = Filterer
    base_test = FiltererTest
