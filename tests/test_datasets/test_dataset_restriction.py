"""Test datasets."""

from __future__ import annotations

import random
from collections.abc import Collection
from typing import NamedTuple

import pytest

from pykeen.datasets.base import Dataset
from pykeen.datasets.utils import get_dataset


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """Fixture for a dataset."""
    return get_dataset(dataset="nations")


def test_restrict_short_cut(dataset: Dataset) -> None:
    """Test trivial dataset restriction."""
    assert dataset.restrict() is dataset


def assert_equal_except_meta(a: Dataset, b: Dataset) -> None:
    """Verify that two datasets are the same except some metadata."""
    assert a is not b
    assert a.num_entities == b.num_entities
    assert a.num_relations == b.num_relations
    assert a.training == b.training
    assert a.testing == b.testing
    assert (a.validation is None) == (b.validation is None)
    assert a.validation == b.validation


def test_trivial(dataset: Dataset) -> None:
    """Test trivial restriction."""
    assert_equal_except_meta(dataset, dataset.restrict(entities=range(dataset.num_entities)))
    assert_equal_except_meta(dataset, dataset.restrict(relations=range(dataset.num_relations)))
    assert_equal_except_meta(dataset, dataset.restrict(entities=[], invert_entity_selection=True))
    assert_equal_except_meta(dataset, dataset.restrict(relations=[], invert_relation_selection=True))


@pytest.fixture()
def rng() -> random.Random:
    """Fixture for random seed."""
    return random.Random(x=42)  # noqa: S311


class PartialCase(NamedTuple):
    """One part of the test case (either entities or relations)."""

    selection: None | Collection[int]
    invert: bool
    max_ids: int


def _make_partial_case(ratio: float, max_id: int, invert: bool, rng: random.Random) -> PartialCase:
    """Make a partial case."""
    if ratio >= 1:
        return PartialCase(selection=None, invert=invert, max_ids=max_id)
    num = int(max_id * ratio)
    all_ids = range(max_id)
    selected = rng.sample(all_ids, k=num)
    if invert:
        return PartialCase(selection=selected, invert=invert, max_ids=max_id - num)
    return PartialCase(selection=selected, invert=invert, max_ids=num)


@pytest.fixture(params=[(0.4, False), (1.0, False), (0.5, True)], ids=str)
def entity_case(request, dataset: Dataset, rng: random.Random) -> PartialCase:
    """Create case part for entities."""
    ratio, invert = request.param
    return _make_partial_case(ratio=ratio, max_id=dataset.num_entities, invert=invert, rng=rng)


@pytest.fixture(params=[(0.4, False), (1.0, False), (0.5, True)], ids=str)
def relation_case(request, dataset: Dataset, rng: random.Random) -> PartialCase:
    """Create case part for relations."""
    ratio, invert = request.param
    return _make_partial_case(ratio=ratio, max_id=dataset.num_relations, invert=invert, rng=rng)


class Case(NamedTuple):
    """Test case for dataset restriction."""

    entities: PartialCase
    relations: PartialCase


@pytest.fixture(ids=str)
def case(entity_case: PartialCase, relation_case: PartialCase) -> Case:
    """Fixture for test cases for dataset restriction."""
    return Case(entities=entity_case, relations=relation_case)


def test_restrict(dataset: Dataset, case: Case) -> None:
    """Test restricting a dataset."""
    # check case
    assert case.entities.max_ids <= dataset.num_entities
    assert case.relations.max_ids <= dataset.num_relations
    # apply restriction
    dataset_restricted = dataset.restrict(
        entities=case.entities.selection,
        relations=case.relations.selection,
        invert_entity_selection=case.entities.invert,
        invert_relation_selection=case.relations.invert,
    )
    # check monotonicity (in counts)
    assert dataset_restricted.num_entities <= case.entities.max_ids
    assert dataset_restricted.num_relations <= case.relations.max_ids
    # check factories
    for key, full in dataset.factory_dict.items():
        restricted = dataset_restricted.factory_dict[key]
        assert restricted.num_triples <= full.num_triples
