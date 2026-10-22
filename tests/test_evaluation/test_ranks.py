"""Test for ranks."""

import pytest
import torch

from pykeen.evaluation.ranks import Ranks

NAN = float("nan")
INF = float("inf")


def test_from_scores_nan_and_inf():
    """Test hand-computed ranks with NaN-filtered and infinite candidate scores."""
    true_score = torch.as_tensor([[1.0], [1.0], [INF], [-INF]])
    all_scores = torch.as_tensor(
        [
            # true score is contained; +inf and 1.0 (tie) are better/equal, NaN is filtered, -inf is worse
            [1.0, INF, 1.0, NAN, -INF, 0.0],
            # only the true score and +inf remain
            [NAN, 1.0, INF, NAN, INF, NAN],
            # +inf true score ties with the other +inf
            [INF, INF, 1.0, -INF, NAN, NAN],
            # -inf true score ties with the other -inf, all finite and +inf are better
            [-INF, -INF, 0.0, NAN, INF, NAN],
        ]
    )
    ranks = Ranks.from_scores(true_score=true_score, all_scores=all_scores)
    assert ranks.number_of_options.tolist() == [5, 3, 4, 4]
    assert ranks.optimistic.tolist() == [2, 3, 1, 3]
    assert ranks.pessimistic.tolist() == [3, 3, 2, 4]
    assert ranks.realistic.tolist() == [2.5, 3.0, 1.5, 3.5]


def test_from_scores_invariant():
    """Test the rank ordering invariant, also for infinite scores."""
    generator = torch.manual_seed(42)
    all_scores = torch.randint(0, 4, size=(50, 12), generator=generator).float()
    all_scores[torch.rand(50, 12, generator=generator) < 0.2] = INF
    all_scores[torch.rand(50, 12, generator=generator) < 0.2] = -INF
    all_scores[torch.rand(50, 12, generator=generator) < 0.2] = NAN
    # the true score is part of the candidates
    all_scores[:, 0] = torch.randint(0, 4, size=(50,), generator=generator).float()
    true_score = all_scores[:, :1].clone()
    ranks = Ranks.from_scores(true_score=true_score, all_scores=all_scores)
    assert (ranks.optimistic >= 1).all()
    assert (ranks.optimistic <= ranks.realistic).all()
    assert (ranks.realistic <= ranks.pessimistic).all()
    assert (ranks.pessimistic <= ranks.number_of_options).all()


def test_from_scores_nan_true_score_warns(caplog: pytest.LogCaptureFixture):
    """Test that a NaN true score emits a warning."""
    with caplog.at_level("WARNING", logger="pykeen.evaluation.ranks"):
        Ranks.from_scores(true_score=torch.as_tensor([[NAN]]), all_scores=torch.as_tensor([[NAN, 1.0]]))
    assert "NaN" in caplog.text


def test_from_scores_finite_true_score_no_warning(caplog: pytest.LogCaptureFixture):
    """Test that no warning is emitted for finite true scores."""
    with caplog.at_level("WARNING", logger="pykeen.evaluation.ranks"):
        Ranks.from_scores(true_score=torch.as_tensor([[1.0]]), all_scores=torch.as_tensor([[1.0, NAN]]))
    assert not caplog.records
