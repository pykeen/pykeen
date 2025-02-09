"""Test for utility methods."""

import pytest

from pykeen.datasets.base import Dataset
from pykeen.datasets.nations import Nations


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """Return a test dataset."""
    return Nations()


def test_merge(dataset: Dataset) -> None:
    """Test merging."""
    tf = dataset.merged()

    # check properties
    assert tf.num_entities == dataset.num_entities
    assert tf.num_relations == dataset.num_relations

    assert tf.num_triples == sum(f.num_triples for f in dataset.factory_dict.values())

    # verify type promotion
    assert isinstance(tf, type(dataset.training))
