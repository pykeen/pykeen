"""Tests for splitting of triples."""
import numpy
import pytest
import torch

from pykeen.triples.splitting import (
    CleanupSplitter,
    CoverageSplitter,
    DeterministicCleaner,
    RandomizedCleaner,
    _get_cover_deterministic,
    get_absolute_split_sizes,
    normalize_ratios,
)
from pykeen.triples.utils import get_entities, get_relations
from pykeen.utils import triple_tensor_to_set
from tests.cases import CleanerTestCase, SplitterTestCase


def test_get_absolute_split_sizes():
    """Test get_absolute_split_sizes."""
    for num_splits, n_total in zip(
        (2, 3, 4),
        (100, 200, 10412),
    ):
        # generate random ratios
        ratios = numpy.random.uniform(size=(num_splits,))
        ratios = ratios / ratios.sum()
        sizes = get_absolute_split_sizes(n_total=n_total, ratios=ratios)
        # check size
        assert len(sizes) == len(ratios)

        # check value range
        assert all(0 <= size <= n_total for size in sizes)

        # check total split
        assert sum(sizes) == n_total

        # check consistency with ratios
        rel_size = numpy.asarray(sizes) / n_total
        # the number of decimal digits equivalent to 1 / n_total
        decimal = numpy.floor(numpy.log10(n_total))
        numpy.testing.assert_almost_equal(rel_size, ratios, decimal=decimal)


def test_normalize_ratios():
    """Test normalize_ratios."""
    for ratios, exp_output in (
        (0.5, (0.5, 0.5)),
        ((0.3, 0.2, 0.4), (0.3, 0.2, 0.4, 0.1)),
        ((0.3, 0.3, 0.4), (0.3, 0.3, 0.4)),
    ):
        output = normalize_ratios(ratios=ratios)
        # check type
        assert isinstance(output, tuple)
        assert all(isinstance(ratio, float) for ratio in output)
        # check values
        assert len(output) >= 2
        assert all(0 <= ratio <= 1 for ratio in output)
        output_np = numpy.asarray(output)
        numpy.testing.assert_almost_equal(output_np.sum(), numpy.ones(1))
        # compare against expected
        numpy.testing.assert_almost_equal(output_np, numpy.asarray(exp_output))


def test_normalize_invalid_ratio():
    """Test invalid ratios."""
    cases = [
        1.1,
        [1.1],
        [0.8, 0.3],
        [0.8, 0.1, 0.2],
    ]
    for ratios in cases:
        with pytest.raises(ValueError):
            _ = normalize_ratios(ratios=ratios)


class DeterministicCleanerTests(CleanerTestCase):
    """Tests for deterministic cleaner."""

    cls = DeterministicCleaner

    def test_manual(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = torch.as_tensor(
            data=[
                [1, 1000, 2],
                [1, 1000, 3],
                [1, 1001, 3],
            ],
            dtype=torch.long,
        )
        testing = torch.as_tensor(
            data=[
                [2, 1001, 3],
                [1, 1002, 4],
            ],
            dtype=torch.long,
        )
        expected_training = torch.as_tensor(
            data=[
                [1, 1000, 2],
                [1, 1000, 3],
                [1, 1001, 3],
                [1, 1002, 4],
            ],
            dtype=torch.long,
        )
        expected_testing = torch.as_tensor(
            data=[
                [2, 1001, 3],
            ],
            dtype=torch.long,
        )

        new_training, new_testing = self.instance.cleanup_pair(training, testing, random_state=...)
        assert (expected_training == new_training).all()
        assert (expected_testing == new_testing).all()


class RandomizedCleanerTests(CleanerTestCase):
    """Tests for randomized cleaner."""

    cls = RandomizedCleaner

    def test_manual(self):
        """Test that triples in a test set can get moved properly to the training set."""
        training = torch.as_tensor(
            data=[
                [1, 1000, 2],
                [1, 1000, 3],
            ],
            dtype=torch.long,
        )
        testing = torch.as_tensor(
            data=[
                [2, 1000, 3],
                [1, 1000, 4],
                [2, 1000, 4],
                [1, 1001, 3],
            ],
            dtype=torch.long,
        )
        expected_training_1 = {
            (1, 1000, 2),
            (1, 1000, 3),
            (1, 1000, 4),
            (1, 1001, 3),
        }
        expected_testing_1 = {
            (2, 1000, 3),
            (2, 1000, 4),
        }

        expected_training_2 = {
            (1, 1000, 2),
            (1, 1000, 3),
            (2, 1000, 4),
            (1, 1001, 3),
        }
        expected_testing_2 = {
            (2, 1000, 3),
            (1, 1000, 4),
        }

        new_training, new_testing = [
            triple_tensor_to_set(arr) for arr in self.instance.cleanup_pair(training, testing, random_state=None)
        ]

        if expected_training_1 == new_training:
            self.assertEqual(expected_testing_1, new_testing)
        elif expected_training_2 == new_training:
            self.assertEqual(expected_testing_2, new_testing)
        else:
            self.fail("training was not correct")


class CleanupSplitterTest(SplitterTestCase):
    """Tests for cleanup splitter."""

    cls = CleanupSplitter


class CoverageSplitterTest(SplitterTestCase):
    """Tests for coverage splitter."""

    cls = CoverageSplitter

    def test_get_cover_deterministic(self):
        """Test _get_cover_deterministic."""
        # generated_triples = generate_triples()
        cover = _get_cover_deterministic(triples=self.mapped_triples)

        # check type
        assert torch.is_tensor(cover)
        assert cover.dtype == torch.bool
        # check format
        assert cover.shape == (self.mapped_triples.shape[0],)

        # check coverage
        self.assertEqual(
            get_entities(self.mapped_triples),
            get_entities(self.mapped_triples[cover]),
            msg="entity coverage is not full",
        )
        self.assertEqual(
            get_relations(self.mapped_triples),
            get_relations(self.mapped_triples[cover]),
            msg="relation coverage is not full",
        )
