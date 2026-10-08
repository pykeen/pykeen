"""Tests for splitting of triples."""

import numpy as np
import pytest
import torch

from pykeen.triples.splitting import (
    CleanupSplitter,
    CoverageSplitter,
    DeterministicCleaner,
    RandomizedCleaner,
    _get_cover_deterministic,
    _prepare_cleanup,
    get_absolute_split_sizes,
    normalize_ratios,
)
from pykeen.triples.utils import get_entities, get_relations
from pykeen.utils import triple_tensor_to_set
from tests.cases import CleanerTestCase, SplitterTestCase


def _generate_rare_triples(
    num_triples: int, num_entities: int, num_relations: int, seed: int
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Generate random triples with a heavy-tailed entity distribution, and split them into two parts."""
    generator = torch.manual_seed(seed)
    # heavy-tailed (Zipf-like) entity distribution => many rare entities
    weights = 1.0 / torch.arange(1, num_entities + 1, dtype=torch.float)
    heads, tails = torch.multinomial(weights, num_samples=2 * num_triples, replacement=True, generator=generator).view(
        2, -1
    )
    relations = torch.randint(num_relations, size=(num_triples,), generator=generator)
    triples = torch.stack([heads, relations, tails], dim=-1).unique(dim=0)
    triples = triples[torch.randperm(triples.shape[0], generator=generator)]
    num_reference = int(0.8 * triples.shape[0])
    return triples[:num_reference], triples[num_reference:]


def _legacy_randomized_cleanup_pair(
    reference: torch.LongTensor, other: torch.LongTensor, generator: torch.Generator
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Clean up by moving one random candidate triple at a time (the original, quadratic implementation)."""
    move_id_mask = _prepare_cleanup(reference, other)
    while move_id_mask.any():
        (candidates,) = move_id_mask.nonzero(as_tuple=True)
        idx = candidates[torch.randint(candidates.shape[0], size=(1,), generator=generator)]
        reference = torch.cat([reference, other[idx].view(1, -1)], dim=0)
        other = torch.cat([other[:idx], other[idx + 1 :]], dim=0)
        move_id_mask = _prepare_cleanup(reference, other)
    return reference, other


def test_get_absolute_split_sizes():
    """Test get_absolute_split_sizes."""
    for num_splits, n_total in zip(
        (2, 3, 4),
        (100, 200, 10412),
        strict=False,
    ):
        # generate random ratios
        rng = np.random.default_rng(seed=42)
        ratios = rng.uniform(size=(num_splits,))
        ratios = ratios / ratios.sum()
        sizes = get_absolute_split_sizes(n_total=n_total, ratios=ratios)
        # check size
        assert len(sizes) == len(ratios)

        # check value range
        assert all(0 <= size <= n_total for size in sizes)

        # check total split
        assert sum(sizes) == n_total

        # check consistency with ratios
        rel_size = np.asarray(sizes) / n_total
        # the number of decimal digits equivalent to 1 / n_total
        decimal = np.floor(np.log10(n_total))
        np.testing.assert_almost_equal(rel_size, ratios, decimal=decimal)


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
        output_np = np.asarray(output)
        np.testing.assert_almost_equal(output_np.sum(), np.ones(1))
        # compare against expected
        np.testing.assert_almost_equal(output_np, np.asarray(exp_output))


def test_normalize_invalid_ratio():
    """Test invalid ratios."""
    cases = [
        1.1,
        [1.1],
        [0.8, 0.3],
        [0.8, 0.1, 0.2],
    ]
    for ratios in cases:
        with pytest.raises(ValueError, match=r"ratios sum to more than 1.0"):
            normalize_ratios(ratios=ratios)


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

        new_training, new_testing = (
            triple_tensor_to_set(arr) for arr in self.instance.cleanup_pair(training, testing, random_state=None)
        )

        if expected_training_1 == new_training:
            assert expected_testing_1 == new_testing
        elif expected_training_2 == new_training:
            assert expected_testing_2 == new_testing
        else:
            self.fail("training was not correct")

    def test_rare_entities(self):
        """Test coverage, reproducibility, and the number of moved triples on random data with rare entities."""
        reference, other = _generate_rare_triples(num_triples=3_000, num_entities=1_000, num_relations=20, seed=0)
        # check for unclean split
        assert not get_entities(other).issubset(get_entities(reference))

        new_reference, new_other = self.instance.cleanup_pair(reference, other, random_state=42)
        # check that no triple got lost, and triples were only moved from other to reference
        assert triple_tensor_to_set(torch.cat([reference, other])) == triple_tensor_to_set(
            torch.cat([new_reference, new_other])
        )
        assert triple_tensor_to_set(reference).issubset(triple_tensor_to_set(new_reference))
        assert triple_tensor_to_set(new_other).issubset(triple_tensor_to_set(other))
        # check coverage
        assert get_entities(new_other).issubset(get_entities(new_reference))
        assert get_relations(new_other).issubset(get_relations(new_reference))

        # check reproducibility
        for x, y in zip(
            (new_reference, new_other), self.instance.cleanup_pair(reference, other, random_state=42), strict=True
        ):
            assert torch.equal(x, y)

        # check number of moved triples: each moved triple covers at least one new ID, and we never move more than the
        # deterministic cleaner
        num_moved = new_reference.shape[0] - reference.shape[0]
        num_uncovered = len(get_entities(other) - get_entities(reference)) + len(
            get_relations(other) - get_relations(reference)
        )
        assert 0 < num_moved <= num_uncovered
        assert num_moved <= int(_prepare_cleanup(reference, other).sum())

        # compare against the original one-triple-at-a-time implementation (same distribution of moved triples)
        num_seeds = 5
        num_moved_new = []
        num_moved_legacy = []
        for seed in range(num_seeds):
            num_moved_new.append(
                self.instance.cleanup_pair(reference, other, random_state=seed)[0].shape[0] - reference.shape[0]
            )
            num_moved_legacy.append(
                _legacy_randomized_cleanup_pair(reference, other, generator=torch.manual_seed(seed))[0].shape[0]
                - reference.shape[0]
            )
        assert np.mean(num_moved_new) <= 1.1 * np.mean(num_moved_legacy)


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
        assert get_entities(self.mapped_triples) == get_entities(self.mapped_triples[cover]), (
            "entity coverage is not full"
        )
        assert get_relations(self.mapped_triples) == get_relations(self.mapped_triples[cover]), (
            "relation coverage is not full"
        )
