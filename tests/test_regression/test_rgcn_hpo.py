"""Regression tests for R-GCN's interaction search space (issue #1568)."""

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.models import RGCN


@pytest.mark.parametrize("interaction", RGCN.hpo_default["interaction"]["choices"])
def test_default_interaction(interaction: str) -> None:
    """Each default HPO interaction must work without interaction-specific kwargs."""
    _check_interaction(interaction=interaction)


def test_explicit_ermlp() -> None:
    """ERMLP remains usable when its embedding dimension is provided explicitly."""
    _check_interaction(interaction="ermlp", interaction_kwargs={"embedding_dim": 4})


def _check_interaction(interaction: str, interaction_kwargs: dict[str, int] | None = None) -> None:
    """Check initialization, scoring, and backpropagation on a small graph."""
    training = Nations().training
    model = RGCN(
        triples_factory=training,
        embedding_dim=4,
        num_layers=1,
        interaction=interaction,
        interaction_kwargs=interaction_kwargs,
        edge_dropout=0.0,
        self_loop_dropout=0.0,
        random_seed=0,
    ).cpu()
    scores = model.score_hrt(training.mapped_triples[:4])
    assert scores.shape == (4, 1)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
